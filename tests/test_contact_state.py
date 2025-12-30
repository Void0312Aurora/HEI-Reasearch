import torch
import unittest
from he_core.state import ContactState

class TestContactState(unittest.TestCase):
    def test_init_and_access(self):
        dim_q = 2
        bs = 3
        state = ContactState(dim_q, bs)
        
        self.assertEqual(state.q.shape, (bs, dim_q))
        self.assertEqual(state.p.shape, (bs, dim_q))
        self.assertEqual(state.s.shape, (bs, 1))
        self.assertEqual(state.flat.shape, (bs, 2*dim_q + 1))
        
        # Test setter
        new_q = torch.ones(bs, dim_q)
        state.q = new_q
        self.assertTrue(torch.all(state.q == 1.0))
        self.assertTrue(torch.all(state.flat[:, :dim_q] == 1.0))
        
        # Test s
        state.s = torch.ones(bs, 1) * 5.0
        self.assertTrue(torch.all(state.s == 5.0))
        self.assertTrue(torch.all(state.flat[:, -1] == 5.0))

if __name__ == '__main__':
    unittest.main()
