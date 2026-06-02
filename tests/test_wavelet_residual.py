import os
import tempfile
import unittest

import torch

from masr.model_utils.conformer.encoder import ConformerEncoder
from masr.model_utils.conformer.wavelet import TemporalHaarWaveletResidual


class WaveletResidualTest(unittest.TestCase):
    def test_shape_even_and_odd(self):
        module = TemporalHaarWaveletResidual(size=16)
        for length in (8, 9):
            x = torch.randn(2, length, 16)
            y = module(x)
            self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_masked_padding_is_unchanged(self):
        module = TemporalHaarWaveletResidual(size=16)
        x = torch.randn(2, 9, 16)
        mask = torch.ones(2, 1, 9, dtype=torch.bool)
        mask[0, :, -3:] = False
        y = module(x, mask)
        self.assertTrue(torch.allclose(y[0, -3:], x[0, -3:], atol=1e-6))

    def test_initial_alpha_and_gate(self):
        module = TemporalHaarWaveletResidual(size=16, alpha_init=-4.0, gate_init=-2.0)
        x = torch.randn(2, 8, 16)
        module(x)
        self.assertAlmostEqual(module.get_alpha_value(), torch.sigmoid(torch.tensor(-4.0)).item(), places=6)
        self.assertAlmostEqual(module.get_gate_mean(), torch.sigmoid(torch.tensor(-2.0)).item(), places=6)

    def test_encoder_disable_state_compatible(self):
        torch.manual_seed(1)
        enc_a = ConformerEncoder(input_size=80, output_size=16, attention_heads=2, linear_units=32,
                                 num_blocks=1, input_layer="conv2d", use_wavelet_residual=False)
        torch.manual_seed(1)
        enc_b = ConformerEncoder(input_size=80, output_size=16, attention_heads=2, linear_units=32,
                                 num_blocks=1, input_layer="conv2d", use_wavelet_residual=False)
        enc_b.load_state_dict(enc_a.state_dict())
        enc_a.eval()
        enc_b.eval()
        x = torch.randn(2, 40, 80)
        lens = torch.tensor([40, 36])
        y_a, m_a = enc_a(x, lens)
        y_b, m_b = enc_b(x, lens)
        self.assertTrue(torch.allclose(y_a, y_b, atol=1e-6))
        self.assertTrue(torch.equal(m_a, m_b))

    def test_encoder_forward_and_chunk(self):
        enc = ConformerEncoder(input_size=80, output_size=16, attention_heads=2, linear_units=32,
                               num_blocks=1, input_layer="conv2d", use_wavelet_residual=True,
                               wavelet_insert_layer=0)
        x = torch.randn(1, 40, 80)
        lens = torch.tensor([40])
        y, mask = enc(x, lens)
        self.assertEqual(y.size(0), 1)
        self.assertEqual(mask.size(0), 1)
        y_chunk, att_cache, cnn_cache = enc.forward_chunk(
            x,
            offset=0,
            required_cache_size=-1,
            att_cache=torch.zeros([0, 0, 0, 0]),
            cnn_cache=torch.zeros([0, 0, 0, 0]),
            att_mask=torch.ones([0, 0, 0], dtype=torch.bool),
        )
        self.assertEqual(y_chunk.size(0), 1)
        self.assertIsNotNone(att_cache)
        self.assertIsNotNone(cnn_cache)

    def test_checkpoint_load_from_old_encoder(self):
        old = ConformerEncoder(input_size=80, output_size=16, attention_heads=2, linear_units=32,
                               num_blocks=1, input_layer="conv2d", use_wavelet_residual=False)
        new = ConformerEncoder(input_size=80, output_size=16, attention_heads=2, linear_units=32,
                               num_blocks=1, input_layer="conv2d", use_wavelet_residual=True,
                               wavelet_insert_layer=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "encoder.pth")
            torch.save(old.state_dict(), path)
            missing, unexpected = new.load_state_dict(torch.load(path), strict=False)
        self.assertTrue(any(k.startswith("wavelet_residual.") for k in missing))
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
