from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from mlbench.data.dummy import build_dummy_module
from mlbench.data.nasa import build_nasa_module


def test_dummy_datamodule_shapes():
    """Test dummy dataset with comprehensive shape and data validation."""
    batch_size = 8
    val_split = 0.25
    seed = 0
    
    module = build_dummy_module(batch_size=batch_size, val_split=val_split, seed=seed)
    train_batch = next(iter(module.train))
    val_batch = next(iter(module.val))

    # Batch size assertions
    assert train_batch.x_seq.shape[0] == batch_size
    assert train_batch.labels["soh_r"].shape[0] == batch_size
    
    # Sequence length assertions (dummy dataset uses t=20 by default)
    assert train_batch.x_seq.shape[1] == 20
    assert val_batch.x_seq.shape[1] == 20
    
    # Feature dimension assertions (dummy dataset uses d=16 by default)
    assert train_batch.x_seq.shape[2] == 16
    assert val_batch.x_seq.shape[2] == 16
    
    # Label shape assertions
    assert train_batch.labels["soh_r"].shape == (batch_size, 2)
    assert val_batch.labels["soh_r"].shape[-1] == 2
    
    # Data type assertions
    assert train_batch.x_seq.dtype == torch.float32
    assert train_batch.labels["soh_r"].dtype == torch.float32
    assert val_batch.x_seq.dtype == torch.float32
    assert val_batch.labels["soh_r"].dtype == torch.float32
    
    # Tensor dimension assertions
    assert train_batch.x_seq.ndim == 3
    assert val_batch.x_seq.ndim == 3
    assert train_batch.labels["soh_r"].ndim == 2
    assert val_batch.labels["soh_r"].ndim == 2
    
    # Times structure assertions
    assert len(train_batch.times) == batch_size
    assert len(val_batch.times) == val_batch.x_seq.shape[0]
    assert all(len(t) == 20 for t in train_batch.times)
    assert all(len(t) == 20 for t in val_batch.times)
    
    # Alpha sequence assertions (should be None for dummy dataset)
    assert train_batch.alpha_seq is None
    assert val_batch.alpha_seq is None


def test_dummy_datamodule_reproducibility():
    """Test that dummy dataset produces consistent data with same seed."""
    batch_size = 4
    seed = 42
    
    # Create two modules with same seed
    module1 = build_dummy_module(batch_size=batch_size, val_split=0.2, seed=seed)
    module2 = build_dummy_module(batch_size=batch_size, val_split=0.2, seed=seed)
    
    # Get first batches
    batch1 = next(iter(module1.train))
    batch2 = next(iter(module2.train))
    
    # Should have same shapes and data types
    assert batch1.x_seq.shape == batch2.x_seq.shape
    assert batch1.labels["soh_r"].shape == batch2.labels["soh_r"].shape
    assert batch1.x_seq.dtype == batch2.x_seq.dtype
    assert batch1.labels["soh_r"].dtype == batch2.labels["soh_r"].dtype
    
    # Data should be finite and reasonable
    assert torch.all(torch.isfinite(batch1.x_seq))
    assert torch.all(torch.isfinite(batch2.x_seq))
    assert torch.all(torch.isfinite(batch1.labels["soh_r"]))
    assert torch.all(torch.isfinite(batch2.labels["soh_r"]))
    
    # Both modules should have same total dataset size
    assert len(module1.train.dataset) == len(module2.train.dataset)
    assert len(module1.val.dataset) == len(module2.val.dataset)


def test_dummy_datamodule_different_seeds():
    """Test that different seeds produce different data."""
    batch_size = 4
    
    module1 = build_dummy_module(batch_size=batch_size, val_split=0.2, seed=0)
    module2 = build_dummy_module(batch_size=batch_size, val_split=0.2, seed=1)
    
    batch1 = next(iter(module1.train))
    batch2 = next(iter(module2.train))
    
    # Should be different
    assert not torch.allclose(batch1.x_seq, batch2.x_seq)
    assert not torch.allclose(batch1.labels["soh_r"], batch2.labels["soh_r"])


def test_dummy_datamodule_edge_cases():
    """Test dummy dataset with edge case parameters."""
    # Test with very small batch size
    module_small = build_dummy_module(batch_size=1, val_split=0.1, seed=0)
    batch_small = next(iter(module_small.train))
    assert batch_small.x_seq.shape[0] == 1
    
    # Test with large validation split
    module_large_val = build_dummy_module(batch_size=4, val_split=0.9, seed=0)
    # Should still have at least one training sample
    train_iter = iter(module_large_val.train)
    batch_large_val = next(train_iter)
    assert batch_large_val.x_seq.shape[0] <= 4
    
    # Test with zero validation split (should still create validation set)
    module_no_val = build_dummy_module(batch_size=4, val_split=0.0, seed=0)
    val_batch = next(iter(module_no_val.val))
    assert val_batch.x_seq.shape[0] >= 1


def test_nasa_datamodule_with_real_data(tmp_path):
    """Test NASA dataset with comprehensive shape and data validation."""
    nasa_root = Path(__file__).resolve().parents[2] / "mlbench" / "data" / "NASA"
    if not nasa_root.exists():
        pytest.skip("NASA dataset not available")

    sample_file = next((f for f in nasa_root.glob("*.mat")), None)
    if sample_file is None:
        pytest.skip("No NASA .mat files found")

    shutil.copy(sample_file, tmp_path / sample_file.name)

    batch_size = 4
    val_split = 0.2
    window = 20
    stride = 5
    seed = 0

    module = build_nasa_module(
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        data_dir=str(tmp_path),
        window=window,
        stride=stride,
    )

    train_batch = next(iter(module.train))
    val_batch = next(iter(module.val))

    # Batch size assertions
    assert train_batch.x_seq.shape[0] == batch_size
    assert train_batch.labels["soh_r"].shape[0] == batch_size
    
    # Sequence length assertions
    assert train_batch.x_seq.shape[1] == window
    assert val_batch.x_seq.shape[1] == window
    
    # Feature dimension assertions (NASA dataset uses 16 features)
    assert train_batch.x_seq.shape[2] == 16
    assert val_batch.x_seq.shape[2] == 16
    
    # Label shape assertions
    assert train_batch.labels["soh_r"].shape == (batch_size, 2)
    assert val_batch.labels["soh_r"].shape[-1] == 2
    
    # Data type assertions
    assert train_batch.x_seq.dtype == torch.float32
    assert train_batch.labels["soh_r"].dtype == torch.float32
    assert val_batch.x_seq.dtype == torch.float32
    assert val_batch.labels["soh_r"].dtype == torch.float32
    
    # Tensor dimension assertions
    assert train_batch.x_seq.ndim == 3
    assert val_batch.x_seq.ndim == 3
    assert train_batch.labels["soh_r"].ndim == 2
    assert val_batch.labels["soh_r"].ndim == 2
    
    # Times structure assertions
    assert len(train_batch.times) == batch_size
    assert len(val_batch.times) == val_batch.x_seq.shape[0]
    assert all(len(t) == window for t in train_batch.times)
    assert all(len(t) == window for t in val_batch.times)
    
    # Alpha sequence assertions (should be None for NASA dataset)
    assert train_batch.alpha_seq is None
    assert val_batch.alpha_seq is None
    
    # SOH value range assertions (SOH should be between 0 and 1)
    assert torch.all(train_batch.labels["soh_r"][:, 0] >= 0.0)
    assert torch.all(train_batch.labels["soh_r"][:, 0] <= 1.0)
    assert torch.all(val_batch.labels["soh_r"][:, 0] >= 0.0)
    assert torch.all(val_batch.labels["soh_r"][:, 0] <= 1.0)
    
    # Feature value assertions (should be finite and reasonable)
    assert torch.all(torch.isfinite(train_batch.x_seq))
    assert torch.all(torch.isfinite(val_batch.x_seq))
    assert torch.all(torch.isfinite(train_batch.labels["soh_r"]))
    assert torch.all(torch.isfinite(val_batch.labels["soh_r"]))


def test_nasa_datamodule_reproducibility(tmp_path):
    """Test that NASA dataset produces consistent data with same seed."""
    nasa_root = Path(__file__).resolve().parents[2] / "mlbench" / "data" / "NASA"
    if not nasa_root.exists():
        pytest.skip("NASA dataset not available")

    sample_file = next((f for f in nasa_root.glob("*.mat")), None)
    if sample_file is None:
        pytest.skip("No NASA .mat files found")

    shutil.copy(sample_file, tmp_path / sample_file.name)

    batch_size = 2
    seed = 42

    # Create two modules with same seed
    module1 = build_nasa_module(
        batch_size=batch_size,
        val_split=0.2,
        seed=seed,
        data_dir=str(tmp_path),
        window=20,
        stride=5,
    )
    module2 = build_nasa_module(
        batch_size=batch_size,
        val_split=0.2,
        seed=seed,
        data_dir=str(tmp_path),
        window=20,
        stride=5,
    )

    # Get first batches
    batch1 = next(iter(module1.train))
    batch2 = next(iter(module2.train))

    # Should have same shapes and data types
    assert batch1.x_seq.shape == batch2.x_seq.shape
    assert batch1.labels["soh_r"].shape == batch2.labels["soh_r"].shape
    assert batch1.x_seq.dtype == batch2.x_seq.dtype
    assert batch1.labels["soh_r"].dtype == batch2.labels["soh_r"].dtype
    
    # Data should be finite and reasonable
    assert torch.all(torch.isfinite(batch1.x_seq))
    assert torch.all(torch.isfinite(batch2.x_seq))
    assert torch.all(torch.isfinite(batch1.labels["soh_r"]))
    assert torch.all(torch.isfinite(batch2.labels["soh_r"]))
    
    # Both modules should have same total dataset size
    assert len(module1.train.dataset) == len(module2.train.dataset)
    assert len(module1.val.dataset) == len(module2.val.dataset)


def test_nasa_datamodule_edge_cases(tmp_path):
    """Test NASA dataset with edge case parameters."""
    nasa_root = Path(__file__).resolve().parents[2] / "mlbench" / "data" / "NASA"
    if not nasa_root.exists():
        pytest.skip("NASA dataset not available")

    sample_file = next((f for f in nasa_root.glob("*.mat")), None)
    if sample_file is None:
        pytest.skip("No NASA .mat files found")

    shutil.copy(sample_file, tmp_path / sample_file.name)

    # Test with very small batch size
    module_small = build_nasa_module(
        batch_size=1,
        val_split=0.1,
        seed=0,
        data_dir=str(tmp_path),
        window=20,
        stride=5,
    )
    batch_small = next(iter(module_small.train))
    assert batch_small.x_seq.shape[0] == 1

    # Test with large validation split
    module_large_val = build_nasa_module(
        batch_size=2,
        val_split=0.9,
        seed=0,
        data_dir=str(tmp_path),
        window=20,
        stride=5,
    )
    # Should still have at least one training sample
    train_iter = iter(module_large_val.train)
    batch_large_val = next(train_iter)
    assert batch_large_val.x_seq.shape[0] <= 2

    # Test with different window sizes
    module_short_window = build_nasa_module(
        batch_size=2,
        val_split=0.2,
        seed=0,
        data_dir=str(tmp_path),
        window=5,
        stride=2,
    )
    batch_short = next(iter(module_short_window.train))
    assert batch_short.x_seq.shape[1] == 5

    # Test with large stride
    module_large_stride = build_nasa_module(
        batch_size=2,
        val_split=0.2,
        seed=0,
        data_dir=str(tmp_path),
        window=20,
        stride=20,
    )
    batch_large_stride = next(iter(module_large_stride.train))
    assert batch_large_stride.x_seq.shape[1] == 20


def test_nasa_datamodule_data_consistency(tmp_path):
    """Test NASA dataset data consistency across batches."""
    nasa_root = Path(__file__).resolve().parents[2] / "mlbench" / "data" / "NASA"
    if not nasa_root.exists():
        pytest.skip("NASA dataset not available")

    sample_file = next((f for f in nasa_root.glob("*.mat")), None)
    if sample_file is None:
        pytest.skip("No NASA .mat files found")

    shutil.copy(sample_file, tmp_path / sample_file.name)

    module = build_nasa_module(
        batch_size=2,
        val_split=0.2,
        seed=0,
        data_dir=str(tmp_path),
        window=20,
        stride=5,
    )

    # Collect multiple batches
    batches = []
    for i, batch in enumerate(module.train):
        batches.append(batch)
        if i >= 2:  # Check first 3 batches
            break

    # All batches should have consistent shapes
    for batch in batches:
        assert batch.x_seq.shape[1] == 20  # window size
        assert batch.x_seq.shape[2] == 16  # feature dimension
        assert batch.labels["soh_r"].shape[1] == 2  # label dimension
        assert batch.x_seq.dtype == torch.float32
        assert batch.labels["soh_r"].dtype == torch.float32

    # SOH values should be reasonable across all batches
    all_soh = torch.cat([batch.labels["soh_r"][:, 0] for batch in batches])
    assert torch.all(all_soh >= 0.0)
    assert torch.all(all_soh <= 1.0)
    assert torch.all(torch.isfinite(all_soh))



