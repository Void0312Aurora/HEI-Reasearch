import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch


@dataclass
class ActiveSamplingConfig:
    wiki_path: str
    clue_path: str
    max_samples_wiki: int = 200000
    max_samples_clue: int = 200000
    hf_dataset: Optional[str] = None
    hf_name: Optional[str] = None
    hf_split: str = "train"
    hf_text_field: str = "text"
    max_samples_hf: int = 0
    min_len: int = 20
    length_bins: Tuple[int, int] = (80, 200)
    source_weights: Dict[str, float] = field(default_factory=lambda: {"wiki": 0.6, "clue": 0.4})
    loss_weight: float = 1.0
    coverage_weight: float = 0.5
    # Learning-progress sampling: prioritize buckets where loss is *decreasing* (i.e. learnable gaps).
    # Computed from the bucket loss EMA; higher progress => higher sampling score.
    progress_weight: float = 0.0
    temperature: float = 1.0
    random_ratio: float = 0.1
    ema_rate: float = 0.1


@dataclass
class TextSample:
    text: str
    source: str
    bucket: str


class CorpusPool:
    def __init__(self, config: ActiveSamplingConfig):
        self.config = config
        self.samples: List[TextSample] = []
        self.bucket_to_indices: Dict[str, List[int]] = {}
        self.bucket_targets: Dict[str, float] = {}
        self._source_bucket_map: Dict[str, List[str]] = {}

    def load(self) -> None:
        self._load_wiki()
        self._load_clue()
        self._load_hf()
        self._build_bucket_indices()
        self._build_bucket_targets()

    def _load_wiki(self) -> None:
        count = 0
        with open(self.config.wiki_path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= self.config.max_samples_wiki:
                    break
                try:
                    doc = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                text = doc.get("text", "")
                if not text:
                    continue
                for para in text.split("\n"):
                    para = para.strip()
                    if len(para) < self.config.min_len:
                        continue
                    bucket = self._bucket_key(para, "wiki")
                    self.samples.append(TextSample(text=para, source="wiki", bucket=bucket))
                    count += 1
                    if count >= self.config.max_samples_wiki:
                        break

    def _load_clue(self) -> None:
        count = 0
        with open(self.config.clue_path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= self.config.max_samples_clue:
                    break
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r"^\d+\s*", "", line)
                if len(line) < self.config.min_len:
                    continue
                bucket = self._bucket_key(line, "clue")
                self.samples.append(TextSample(text=line, source="clue", bucket=bucket))
                count += 1

    def _load_hf(self) -> None:
        if not self.config.hf_dataset or self.config.max_samples_hf <= 0:
            return

        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "HuggingFace streaming requested but `datasets` is not installed. "
                "Install it with: `pip install datasets`."
            ) from exc

        count = 0
        ds_kwargs = {}
        if self.config.hf_name:
            ds_kwargs["name"] = self.config.hf_name

        dataset = load_dataset(
            self.config.hf_dataset,
            **ds_kwargs,
            split=self.config.hf_split,
            streaming=True,
        )

        text_field = self.config.hf_text_field
        for row in dataset:
            if count >= self.config.max_samples_hf:
                break
            text = row.get(text_field, None)
            if text is None:
                continue
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            # Keep paragraph semantics similar to wiki json ("text" contains newlines).
            for para in text.split("\n"):
                para = para.strip()
                if len(para) < self.config.min_len:
                    continue
                bucket = self._bucket_key(para, "hf")
                self.samples.append(TextSample(text=para, source="hf", bucket=bucket))
                count += 1
                if count >= self.config.max_samples_hf:
                    break

    def _bucket_key(self, text: str, source: str) -> str:
        length = len(text)
        if length <= self.config.length_bins[0]:
            length_bin = "short"
        elif length <= self.config.length_bins[1]:
            length_bin = "medium"
        else:
            length_bin = "long"

        has_question = "question" if ("?" in text or "？" in text) else "statement"
        return f"{source}:{length_bin}:{has_question}"

    def _build_bucket_indices(self) -> None:
        for idx, sample in enumerate(self.samples):
            self.bucket_to_indices.setdefault(sample.bucket, []).append(idx)

    def _build_bucket_targets(self) -> None:
        self._source_bucket_map.clear()

        for bucket in self.bucket_to_indices:
            source = bucket.split(":", 1)[0]
            self._source_bucket_map.setdefault(source, []).append(bucket)

        available_sources = {
            source: buckets for source, buckets in self._source_bucket_map.items() if buckets
        }
        source_weights = self._normalize_source_weights({
            k: v for k, v in self.config.source_weights.items() if k in available_sources
        }) if available_sources else {}

        # Target distribution should reflect the environment's empirical distribution.
        # The previous "uniform per bucket" rule can massively oversample rare buckets
        # (e.g. wiki:*:question), causing heavy repetition/overfit and poor generalization.
        for source, buckets in available_sources.items():
            weight = source_weights.get(source, 0.0)
            if not buckets or weight <= 0.0:
                continue
            counts = {bucket: len(self.bucket_to_indices.get(bucket, [])) for bucket in buckets}
            total = sum(counts.values())
            if total <= 0:
                continue
            for bucket in buckets:
                self.bucket_targets[bucket] = weight * (counts[bucket] / total)

        if not self.bucket_targets and self.bucket_to_indices:
            # Fallback: uniform targets across buckets when weights are unavailable.
            per_bucket = 1.0 / len(self.bucket_to_indices)
            for bucket in self.bucket_to_indices:
                self.bucket_targets[bucket] = per_bucket

    @staticmethod
    def _normalize_source_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}


class ActiveSampler:
    def __init__(self, pool: CorpusPool, config: ActiveSamplingConfig):
        self.pool = pool
        self.config = config
        self.bucket_counts: Dict[str, int] = {k: 0 for k in pool.bucket_to_indices}
        self.bucket_losses: Dict[str, float] = {k: 0.0 for k in pool.bucket_to_indices}
        self.bucket_progress: Dict[str, float] = {k: 0.0 for k in pool.bucket_to_indices}
        self.total_seen = 0
        self.token_cache: Optional[torch.Tensor] = None
        self.mask_cache: Optional[torch.Tensor] = None

    def sample_indices(self, batch_size: int) -> Tuple[List[int], List[str]]:
        """Sample indices + their bucket keys.

        Always returns exactly `batch_size` indices when the pool is non-empty.
        """
        if not self.pool.samples or batch_size <= 0:
            return [], []

        random_count = int(batch_size * self.config.random_ratio)
        active_count = batch_size - random_count

        indices = []
        buckets = []

        if random_count > 0:
            for _ in range(random_count):
                idx = random.randrange(len(self.pool.samples))
                indices.append(idx)
                buckets.append(self.pool.samples[idx].bucket)

        if active_count > 0:
            bucket_choices = self._sample_buckets(active_count)
            if not bucket_choices:
                # Fallback: no valid bucket scores yet; sample uniformly.
                bucket_choices = [self.pool.samples[random.randrange(len(self.pool.samples))].bucket for _ in range(active_count)]
            for bucket in bucket_choices:
                candidates = self.pool.bucket_to_indices[bucket]
                if not candidates:
                    idx = random.randrange(len(self.pool.samples))
                    bucket = self.pool.samples[idx].bucket
                else:
                    idx = random.choice(candidates)
                indices.append(idx)
                buckets.append(bucket)

        # Safety: guarantee fixed batch size for CUDA-graph workflows.
        while len(indices) < batch_size:
            idx = random.randrange(len(self.pool.samples))
            indices.append(idx)
            buckets.append(self.pool.samples[idx].bucket)

        if len(indices) > batch_size:
            indices = indices[:batch_size]
            buckets = buckets[:batch_size]

        return indices, buckets

    def sample_batch(
        self,
        batch_size: int,
        *,
        with_texts: bool = True,
    ) -> Tuple[List[str], List[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.pool.samples:
            return [], [], None, None

        indices, buckets = self.sample_indices(batch_size)

        texts: List[str]
        if with_texts:
            texts = [self.pool.samples[i].text for i in indices]
        else:
            texts = []
        token_batch = None
        mask_batch = None
        if self.token_cache is not None and self.mask_cache is not None:
            idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.token_cache.device)
            token_batch = self.token_cache.index_select(0, idx_tensor)
            mask_batch = self.mask_cache.index_select(0, idx_tensor)
        return texts, buckets, token_batch, mask_batch

    def _sample_buckets(self, count: int) -> List[str]:
        scores = self._bucket_scores()
        if not scores:
            return []

        buckets = list(scores.keys())
        weights = list(scores.values())
        temp = float(getattr(self.config, "temperature", 1.0) or 1.0)
        if (not math.isfinite(temp)) or temp <= 0:
            temp = 1.0
        if abs(temp - 1.0) > 1e-6:
            inv_t = 1.0 / temp
            weights = [max(w, 1e-12) ** inv_t for w in weights]
        total = sum(weights)
        if (not math.isfinite(total)) or total <= 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total for w in weights]
            if not all(math.isfinite(w) and w >= 0.0 for w in weights):
                weights = [1.0 / len(weights)] * len(weights)

        return random.choices(buckets, weights=weights, k=count)

    def _bucket_scores(self) -> Dict[str, float]:
        scores = {}
        total_seen = max(self.total_seen, 1)
        avg_loss = self._global_avg_loss()
        if (not math.isfinite(avg_loss)) or avg_loss <= 0:
            avg_loss = 1.0
        avg_progress = self._global_avg_progress()
        if (not math.isfinite(avg_progress)) or avg_progress <= 0:
            avg_progress = 0.0

        for bucket, target_ratio in self.pool.bucket_targets.items():
            count = self.bucket_counts.get(bucket, 0)
            loss = self.bucket_losses.get(bucket, avg_loss)
            if not math.isfinite(loss):
                loss = avg_loss

            loss_norm = loss / (avg_loss + 1e-6)
            if not math.isfinite(loss_norm):
                loss_norm = 1.0
            expected = target_ratio * total_seen
            deficit = max(expected - count, 0.0)
            deficit_norm = deficit / (total_seen + 1e-6)

            progress = self.bucket_progress.get(bucket, 0.0)
            if not math.isfinite(progress):
                progress = 0.0
            progress = max(progress, 0.0)
            progress_norm = (progress / (avg_progress + 1e-6)) if avg_progress > 0 else 0.0

            score = (
                self.config.loss_weight * loss_norm
                + self.config.coverage_weight * deficit_norm
                + float(getattr(self.config, "progress_weight", 0.0) or 0.0) * progress_norm
            )
            if not math.isfinite(score):
                score = 1e-6
            scores[bucket] = max(score, 1e-6)

        return scores

    def _global_avg_loss(self) -> float:
        total = 0.0
        count = 0
        for bucket, loss in self.bucket_losses.items():
            if self.bucket_counts.get(bucket, 0) > 0:
                if math.isfinite(loss):
                    total += loss
                    count += 1
        return total / count if count > 0 else 1.0

    def _global_avg_progress(self) -> float:
        total = 0.0
        count = 0
        for bucket, prog in self.bucket_progress.items():
            if self.bucket_counts.get(bucket, 0) > 0:
                if math.isfinite(prog) and prog > 0:
                    total += prog
                    count += 1
        return total / count if count > 0 else 0.0

    def update_stats(self, bucket_keys: List[str], loss_values) -> None:
        # Backward-compatible wrapper: update both counts and loss EMA.
        self.update_counts(bucket_keys)
        self.update_losses(bucket_keys, loss_values)

    def update_counts(self, bucket_keys: List[str]) -> None:
        if not bucket_keys:
            return
        for bucket in bucket_keys:
            self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1
            self.total_seen += 1

    def update_losses(self, bucket_keys: List[str], loss_values) -> None:
        if not bucket_keys:
            return
        if isinstance(loss_values, (float, int)):
            loss_values = [float(loss_values)] * len(bucket_keys)
        elif hasattr(loss_values, "tolist"):
            loss_values = loss_values.tolist()

        for bucket, loss_value in zip(bucket_keys, loss_values):
            try:
                loss_f = float(loss_value)
            except Exception:
                loss_f = float("nan")

            prev = self.bucket_losses.get(bucket, loss_f)
            if not math.isfinite(prev):
                prev = 1.0
            if math.isfinite(loss_f):
                new = (1.0 - self.config.ema_rate) * prev + self.config.ema_rate * loss_f
                if math.isfinite(new):
                    self.bucket_losses[bucket] = new
                    # Learning progress: positive improvement in the loss EMA.
                    # We keep a separate EMA of progress to reduce sampling oscillations.
                    prog = max(prev - new, 0.0)
                    prev_prog = self.bucket_progress.get(bucket, 0.0)
                    if not math.isfinite(prev_prog):
                        prev_prog = 0.0
                    prog_ema = (1.0 - self.config.ema_rate) * prev_prog + self.config.ema_rate * prog
                    if math.isfinite(prog_ema):
                        self.bucket_progress[bucket] = prog_ema
                else:
                    # Keep a finite value to avoid poisoning subsequent sampling stats.
                    self.bucket_losses[bucket] = prev
            else:
                self.bucket_losses[bucket] = prev

    def set_token_cache(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if token_ids.shape != attention_mask.shape:
            raise ValueError("Token cache and mask cache must share the same shape.")
        self.token_cache = token_ids
        self.mask_cache = attention_mask
