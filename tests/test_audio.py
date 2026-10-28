import numpy as np

from vaice.audio import from_pcm16, to_mono_float, to_pcm16


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


def test_pcm16_is_mono_little_endian_at_the_requested_rate():
    tone = (10000 * np.sin(2 * np.pi * 440 * np.arange(4800) / 48000)).astype(np.int16)
    out = np.frombuffer(to_pcm16((48000, np.stack([tone, tone])), 16000), dtype="<i2")
    assert len(out) == 1600
    expected = 10000 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)
    np.testing.assert_allclose(out[100:-100], expected[100:-100], atol=150)


def test_pcm16_at_the_same_rate_is_not_resampled():
    out = np.frombuffer(to_pcm16((16000, np.array([[0, 100, -100]], dtype=np.int16)), 16000), dtype="<i2")
    assert out.tolist() == [0, 100, -100]


def test_pcm16_clips_instead_of_wrapping():
    loud = np.array([[0.0, 1.5, -1.5]], dtype=np.float32)
    out = np.frombuffer(to_pcm16((16000, loud), 16000), dtype="<i2")
    assert out.tolist() == [0, 32767, -32768]


def test_pcm16_round_trips_into_float32():
    out = from_pcm16(np.array([0, 16384, -32768], dtype="<i2").tobytes())
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [0.0, 0.5, -1.0])
