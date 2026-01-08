import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

import torch

from he_core.language_interface import SimpleTokenizer, ByteTokenizer
from training.soul_language_training import TrainingConfig, SoulLanguageTrainer

T = TypeVar("T")


def _filter_dataclass_kwargs(dataclass_type: Type[T], values: Dict[str, Any]) -> Dict[str, Any]:
    allowed = getattr(dataclass_type, "__dataclass_fields__", {})
    return {k: v for k, v in values.items() if k in allowed}


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tokenizer(save_dir: str, tokenizer: SimpleTokenizer, filename: str = "tokenizer.json") -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    tokenizer.save(path)
    return path


def load_tokenizer(path: str) -> SimpleTokenizer:
    data = load_json(path)
    tok_type = str(data.get("type", "")).lower()
    mode = str(data.get("mode", "")).lower()
    if tok_type == "byte" or mode == "byte":
        tokenizer = ByteTokenizer()
        tokenizer.load(path)
        return tokenizer  # type: ignore[return-value]
    tokenizer = SimpleTokenizer(vocab_size=1, mode="char")
    tokenizer.load(path)
    return tokenizer


def save_training_config(save_dir: str, config: TrainingConfig, filename: str = "train_config.json") -> str:
    os.makedirs(save_dir, exist_ok=True)
    payload = asdict(config) if is_dataclass(config) else dict(config)
    path = os.path.join(save_dir, filename)
    save_json(path, payload)
    return path


def save_checkpoint(
    save_dir: str,
    *,
    trainer: SoulLanguageTrainer,
    tokenizer: SimpleTokenizer,
    train_cfg: TrainingConfig,
    step: int,
    total_tokens: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
    keep_every: int = 0,
) -> Tuple[str, str]:
    """
    Save a self-contained checkpoint bundle under `save_dir`.

    Files:
    - tokenizer.json
    - train_config.json
    - last.pt
    - step{step}_tok{total_tokens}.pt (optional, controlled by keep_every)
    """
    os.makedirs(save_dir, exist_ok=True)
    save_tokenizer(save_dir, tokenizer)
    save_training_config(save_dir, train_cfg)

    tok_type = getattr(tokenizer, "type", None) or ("byte" if getattr(tokenizer, "mode", "") == "byte" else "simple")
    payload: Dict[str, Any] = {
        "model_state": trainer.state_dict(),
        "train_cfg": asdict(train_cfg),
        "tokenizer_state": {
            "type": str(tok_type),
            "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
            "mode": getattr(tokenizer, "mode", "char"),
            "special_tokens": getattr(tokenizer, "special_tokens", None),
            "byte_offset": getattr(tokenizer, "byte_offset", None),
            "token_to_id": getattr(tokenizer, "token_to_id", {}),
        },
        "step": int(step),
        "total_tokens": int(total_tokens),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra

    last_path = os.path.join(save_dir, "last.pt")
    torch.save(payload, last_path)

    kept_path = ""
    if keep_every and step % keep_every == 0:
        kept_name = f"step{step:07d}_tok{total_tokens}.pt"
        kept_path = os.path.join(save_dir, kept_name)
        torch.save(payload, kept_path)

    return last_path, kept_path


def load_trainer_from_checkpoint(
    ckpt_path: str,
    *,
    device: str,
    tokenizer_path: Optional[str] = None,
    override_cfg: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> Tuple[SoulLanguageTrainer, SimpleTokenizer, TrainingConfig, Dict[str, Any]]:
    """
    Load tokenizer + config + model weights and construct a ready-to-run trainer.
    """
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    if tokenizer_path is None:
        tokenizer_path = os.path.join(ckpt_dir, "tokenizer.json")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    tokenizer: Optional[SimpleTokenizer] = None
    if os.path.exists(tokenizer_path):
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        tok_state = ckpt.get("tokenizer_state", None)
        if isinstance(tok_state, dict) and isinstance(tok_state.get("token_to_id", None), dict):
            tok_type = str(tok_state.get("type", "")).lower()
            mode = str(tok_state.get("mode", "char")).lower()
            if tok_type == "byte" or mode == "byte":
                special_tokens = tok_state.get("special_tokens", None) or ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
                tokenizer = ByteTokenizer(special_tokens=list(special_tokens))
            else:
                special_tokens = tok_state.get("special_tokens", None) or ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']
                tokenizer = SimpleTokenizer(
                    vocab_size=int(tok_state.get("vocab_size", 1)),
                    mode=str(tok_state.get("mode", "char")),
                    special_tokens=list(special_tokens),
                )
                tokenizer.token_to_id = dict(tok_state["token_to_id"])
                tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
                tokenizer.pad_id = tokenizer.token_to_id.get("<PAD>", 0)
                tokenizer.unk_id = tokenizer.token_to_id.get("<UNK>", 1)
                tokenizer.bos_id = tokenizer.token_to_id.get("<BOS>", 2)
                tokenizer.eos_id = tokenizer.token_to_id.get("<EOS>", 3)
        else:
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}. "
                f"Provide --tokenizer_path or re-train with --save_dir to generate tokenizer.json."
            )

    cfg_dict = ckpt.get("train_cfg", {})
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}

    cfg_dict = dict(cfg_dict)
    cfg_dict["device"] = device
    override_cfg = override_cfg or {}
    cfg_dict.update(override_cfg)

    cfg_filtered = _filter_dataclass_kwargs(TrainingConfig, cfg_dict)
    train_cfg = TrainingConfig(**cfg_filtered)

    train_cfg.vocab_size = len(tokenizer)

    trainer = SoulLanguageTrainer(train_cfg, tokenizer).to(device)
    trainer.load_state_dict(ckpt["model_state"], strict=strict)

    return trainer, tokenizer, train_cfg, ckpt
