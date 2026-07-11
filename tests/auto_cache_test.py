"""
Tests for the autotuning cache's own logic: shape-key derivation and
JSON serialization round-tripping. Deliberately CPU-only and independent
of CUDA -- these run on every CI job regardless of whether the runner has
a GPU, unlike tests/test_correctness.py.
"""

import json

from awq_fast_dequant.autotune import _shape_key


class FakeTensor:
    """
    Minimal stand-in exposing only the .shape attribute _shape_key reads,
    so this test doesn't need a real CUDA tensor -- keeping it runnable
    on CPU-only CI runners.
    """
    def __init__(self, shape):
        self.shape = shape


def test_shape_key_extracts_correct_dimensions():
    # qweight: [K, M/8], scales: [K/group_size, M]
    qweight = FakeTensor((1536, 1120))   # K=1536, M/8=1120 -> M=8960
    scales = FakeTensor((12, 8960))      # K/group_size=12 -> group_size=128
    key = _shape_key(qweight, scales)
    assert key == (1536, 8960, 128)


def test_shape_key_differs_for_different_shapes():
    qweight_a = FakeTensor((1536, 1120))
    scales_a = FakeTensor((12, 8960))
    qweight_b = FakeTensor((8960, 192))
    scales_b = FakeTensor((70, 1536))
    assert _shape_key(qweight_a, scales_a) != _shape_key(qweight_b, scales_b)


def test_cache_json_round_trip(tmp_path, monkeypatch):
    """
    The cache serializes tuple keys as comma-joined strings (JSON requires
    string keys) and must parse them back into the same tuples on load.
    This tests that round trip directly, using a temporary cache location
    so it doesn't touch the real ~/.cache/awq_fast_dequant/ file.
    """
    import awq_fast_dequant.autotune as autotune_module

    fake_cache_dir = tmp_path / "awq_fast_dequant"

    def fake_cache_path():
        # Mirrors the real _cache_path()'s behavior of ensuring the
        # directory exists before returning the file path.
        fake_cache_dir.mkdir(parents=True, exist_ok=True)
        return str(fake_cache_dir / "block_size_cache.json")

    monkeypatch.setattr(autotune_module, "_cache_path", fake_cache_path)

    original_cache = {(1536, 8960, 128): 64, (8960, 1536, 128): 128}
    autotune_module._block_size_cache.clear()
    autotune_module._block_size_cache.update(original_cache)
    autotune_module._save_cache_to_disk()

    cache_file = fake_cache_dir / "block_size_cache.json"
    assert cache_file.exists()

    with open(cache_file) as f:
        raw = json.load(f)
    assert all(isinstance(k, str) for k in raw.keys())

    reloaded = autotune_module._load_cache_from_disk()
    assert reloaded == original_cache

    autotune_module._block_size_cache.clear()