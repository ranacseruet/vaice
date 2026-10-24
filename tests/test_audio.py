import numpy as np

from vaice.audio import to_mono_float


def test_int16_channels_first_becomes_float_mono():
    out = to_mono_float(np.array([[0, 16384, -32768]], dtype=np.int16))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.5, -1.0])


def test_stereo_is_averaged():
    out = to_mono_float(np.array([[1000, 1000], [3000, 3000]], dtype=np.int16))
    np.testing.assert_allclose(out, [2000 / 32768] * 2)


def test_float_input_passes_through():
    out = to_mono_float(np.array([0.25, -0.5], dtype=np.float32))
    np.testing.assert_allclose(out, [0.25, -0.5])
