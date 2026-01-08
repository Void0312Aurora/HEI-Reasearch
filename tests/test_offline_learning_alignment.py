import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from he_core.soul_entity import create_soul_entity
from he_core.state import ContactState
from he_core.language_interface import SimpleTokenizer
from training.soul_language_training import TrainingConfig, SoulLanguageTrainer


class TestOfflineLearningAlignment(unittest.TestCase):
    def test_soul_entity_forward_tensor_is_differentiable(self):
        torch.manual_seed(0)
        dim_q = 8
        batch_size = 2

        entity = create_soul_entity({
            "dim_q": dim_q,
            "dim_u": dim_q,
            "dim_z": 4,
            "num_charts": 3,
            "stiffness": 0.01,
        }).to("cpu")

        state_flat = torch.randn(batch_size, 2 * dim_q + 1, requires_grad=True)
        state = ContactState(dim_q, batch_size, device="cpu", flat_tensor=state_flat)

        prev_weights = entity.atlas.router(state.q.clone())
        out = entity.forward_tensor(
            state_flat=state.flat,
            u_dict={},
            dt=0.05,
            prev_chart_weights=prev_weights,
            detach_next_prev_weights=False,
        )

        loss = out["free_energy"]
        loss.backward()

        self.assertIsNotNone(state_flat.grad)
        self.assertGreater(state_flat.grad.abs().sum().item(), 0.0)

    def test_trainer_offline_rollout_backprops(self):
        torch.manual_seed(0)

        tokenizer = SimpleTokenizer(vocab_size=200, mode="char")
        tokenizer.build_vocab(["这是一个测试。", "自由能驱动离线学习。"] * 20)

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
        trainer.eval()  # disable dropout for determinism

        ids1 = tokenizer.encode("这是一个测试。")
        ids2 = tokenizer.encode("自由能驱动离线学习。")
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
        loss.backward()

        has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.parameters())
        self.assertTrue(has_grad)


if __name__ == "__main__":
    unittest.main()
