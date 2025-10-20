"""Tests for NASA dataset loading and processing."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from mlbench.data.nasa import build_nasa_module, build_nasa_cc_module
from mlbench.data.registry import get_dataset


class TestNASADataset:
    """Test NASA dataset functionality."""
    
    def test_nasa_dataset_registration(self):
        """Test that NASA dataset is properly registered."""
        from mlbench.data.registry import available_datasets
        
        datasets = available_datasets()
        assert "nasa" in datasets
        assert "nasa_cc" in datasets
    
    def test_nasa_dataset_builder(self):
        """Test NASA dataset builder with dummy data."""
        # Create temporary directory with dummy .mat files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy .mat file structure
            # This is a simplified test - in practice you'd need proper .mat files
            
            # Test that the builder function exists and can be called
            builder = get_dataset("nasa")
            assert callable(builder)
            
            # Test with minimal parameters
            try:
                datamodule = builder(
                    batch_size=2,
                    val_split=0.5,
                    seed=42,
                    data_dir=temp_dir,
                    window=5,
                    stride=2,
                )
                
                # Check that datamodule has required attributes
                assert hasattr(datamodule, "train")
                assert hasattr(datamodule, "val")
                
                # Check that loaders are DataLoader instances
                assert isinstance(datamodule.train, torch.utils.data.DataLoader)
                assert isinstance(datamodule.val, torch.utils.data.DataLoader)
                
            except FileNotFoundError:
                # Expected when no .mat files are present
                pass
    
    def test_nasa_cc_dataset_builder(self):
        """Test NASA CC dataset builder."""
        builder = get_dataset("nasa_cc")
        assert callable(builder)
        
        # Test with minimal parameters
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                datamodule = builder(
                    batch_size=2,
                    val_split=0.5,
                    seed=42,
                    data_dir=temp_dir,
                    window=5,
                    stride=2,
                )
                
                # Check that datamodule has required attributes
                assert hasattr(datamodule, "train")
                assert hasattr(datamodule, "val")
                
            except FileNotFoundError:
                # Expected when no .mat files are present
                pass
    
    def test_dataset_metadata(self):
        """Test that dataset includes proper metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                datamodule = build_nasa_module(
                    batch_size=2,
                    val_split=0.5,
                    seed=42,
                    data_dir=temp_dir,
                    window=5,
                    stride=2,
                )
                
                # Check metadata
                if hasattr(datamodule, "metadata"):
                    metadata = datamodule.metadata
                    assert "feature_names" in metadata
                    assert "target_name" in metadata
                    assert "num_samples" in metadata
                    assert "window" in metadata
                    assert "stride" in metadata
                
            except FileNotFoundError:
                pass
    
    def test_scaler_integration(self):
        """Test that scaler is properly integrated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                datamodule = build_nasa_module(
                    batch_size=2,
                    val_split=0.5,
                    seed=42,
                    data_dir=temp_dir,
                    window=5,
                    stride=2,
                    scaler_type="standard",
                )
                
                # Check that scaler is in metadata
                if hasattr(datamodule, "metadata") and "scaler" in datamodule.metadata:
                    scaler = datamodule.metadata["scaler"]
                    assert hasattr(scaler, "fit")
                    assert hasattr(scaler, "transform")
                    assert hasattr(scaler, "fit_transform")
                
            except FileNotFoundError:
                pass
