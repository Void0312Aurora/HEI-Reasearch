import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from he_core.language_interface import SimpleTokenizer
from training.checkpoint_io import load_trainer_from_checkpoint, save_checkpoint
from training.soul_language_training import TrainingConfig, SoulLanguageTrainer


class TestCheckpointIO(unittest.TestCase):
    def test_save_and_load_checkpoint_bundle(self):
        torch.manual_seed(0)

        tokenizer = SimpleTokenizer(vocab_size=200, mode="char")
        tokenizer.build_vocab(["这是一个测试。", "离线学习应当可微。"] * 20)

        cfg = TrainingConfig(
            dim_q=16,
            dim_embed=32,
            vocab_size=len(tokenizer),
            batch_size=2,
            max_seq_len=32,
            device="cpu",
            num_evolution_steps=1,
            num_offline_steps=1,
            offline_dt=0.05,
            offline_replay_mode="random",
            offline_weight=1.0,
            offline_detach_init=False,
            reset_each_batch=True,
        )
        trainer = SoulLanguageTrainer(cfg, tokenizer).to("cpu")
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-4)

        ids1 = tokenizer.encode("这是一个测试。")
        ids2 = tokenizer.encode("离线学习应当可微。")
        local_len = min(max(len(ids1), len(ids2)), cfg.max_seq_len)

        def pad_to(ids, length):
            ids = ids[:length]
            mask = [1] * len(ids)
            if len(ids) < length:
                pad_n = length - len(ids)
                ids = ids + [tokenizer.pad_id] * pad_n
                mask = mask + [0] * pad_n
            return ids, mask

        p1, m1 = pad_to(ids1, local_len)
        p2, m2 = pad_to(ids2, local_len)
        batch = {
            "input_ids": torch.tensor([p1, p2], dtype=torch.long),
            "attention_mask": torch.tensor([m1, m2], dtype=torch.long),
        }

        out = trainer.train_step(batch)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            last_path, kept_path = save_checkpoint(
                tmpdir,
                trainer=trainer,
                tokenizer=tokenizer,
                train_cfg=cfg,
                step=1,
                total_tokens=int(batch["attention_mask"].sum().item()),
                optimizer=optimizer,
                keep_every=1,
            )

            self.assertTrue(os.path.exists(last_path))
            self.assertTrue(os.path.exists(kept_path))

            loaded_trainer, loaded_tokenizer, loaded_cfg, meta = load_trainer_from_checkpoint(
                last_path,
                device="cpu",
                strict=False,
            )

            self.assertEqual(len(loaded_tokenizer), len(tokenizer))
            self.assertEqual(loaded_cfg.dim_q, cfg.dim_q)

            out2 = loaded_trainer.train_step(batch)
            self.assertTrue(torch.isfinite(out2["free_energy"]).all())


if __name__ == "__main__":
    unittest.main()

