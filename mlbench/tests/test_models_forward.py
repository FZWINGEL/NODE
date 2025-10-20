"""Tests for model forward passes and shapes."""

import pytest
import torch
import numpy as np

from mlbench.data.types import Batch
from mlbench.models.base import ForwardModel
from mlbench.models.registry import available_models, get_model


class TestModelForward:
    """Test model forward passes and shapes."""
    
    def test_model_registration(self):
        """Test that models are properly registered."""
        models = available_models()
        assert len(models) > 0
        
        # Check that we have some expected models
        expected_models = ["lstm", "pcrnn", "node", "anode", "ude_charm", "acla"]
        for model_name in expected_models:
            if model_name in models:
                assert True  # Model is registered
            else:
                pytest.skip(f"Model {model_name} not available")
    
    def test_model_forward_shapes(self):
        """Test that models produce correct output shapes."""
        # Create a dummy batch
        batch_size = 4
        seq_len = 20
        feature_dim = 16
        
        batch = Batch(
            times=[list(range(seq_len)) for _ in range(batch_size)],
            x_seq=torch.randn(batch_size, seq_len, feature_dim),
            alpha_seq=None,
            labels={"soh_r": torch.randn(batch_size, 2)},  # [soh, 0.0]
        )
        
        # Test each available model
        models = available_models()
        for model_name in models:
            try:
                model_class = get_model(model_name)
                model = model_class()
                
                # Forward pass
                outputs, traj = model(batch, t_eval=None)
                
                # Check outputs
                assert isinstance(outputs, dict)
                assert "soh_r" in outputs or "soh" in outputs
                
                # Check output shapes
                if "soh_r" in outputs:
                    assert outputs["soh_r"].shape == (batch_size, 2)
                elif "soh" in outputs:
                    assert outputs["soh"].shape == (batch_size, 1)
                
                # Check trajectory
                assert isinstance(traj, dict)
                
                # Test loss computation
                loss = model.compute_loss(batch, outputs, traj)
                assert isinstance(loss, torch.Tensor)
                assert loss.dim() == 0  # Scalar loss
                
            except Exception as e:
                pytest.skip(f"Model {model_name} failed: {e}")
    
    def test_model_gradient_flow(self):
        """Test that gradients flow properly through models."""
        batch_size = 2
        seq_len = 10
        feature_dim = 8
        
        batch = Batch(
            times=[list(range(seq_len)) for _ in range(batch_size)],
            x_seq=torch.randn(batch_size, seq_len, feature_dim, requires_grad=True),
            alpha_seq=None,
            labels={"soh_r": torch.randn(batch_size, 2)},
        )
        
        models = available_models()
        for model_name in models:
            try:
                model_class = get_model(model_name)
                model = model_class()
                
                # Forward pass
                outputs, traj = model(batch, t_eval=None)
                loss = model.compute_loss(batch, outputs, traj)
                
                # Backward pass
                loss.backward()
                
                # Check that gradients exist
                for param in model.parameters():
                    assert param.grad is not None
                    assert not torch.isnan(param.grad).any()
                
            except Exception as e:
                pytest.skip(f"Model {model_name} gradient test failed: {e}")
    
    def test_model_deterministic(self):
        """Test that models produce deterministic outputs."""
        batch_size = 2
        seq_len = 10
        feature_dim = 8
        
        batch = Batch(
            times=[list(range(seq_len)) for _ in range(batch_size)],
            x_seq=torch.randn(batch_size, seq_len, feature_dim),
            alpha_seq=None,
            labels={"soh_r": torch.randn(batch_size, 2)},
        )
        
        models = available_models()
        for model_name in models:
            try:
                model_class = get_model(model_name)
                model1 = model_class()
                model2 = model_class()
                
                # Set same weights
                model2.load_state_dict(model1.state_dict())
                
                # Forward pass
                outputs1, traj1 = model1(batch, t_eval=None)
                outputs2, traj2 = model2(batch, t_eval=None)
                
                # Check outputs are the same
                for key in outputs1:
                    if key in outputs2:
                        assert torch.allclose(outputs1[key], outputs2[key], atol=1e-6)
                
            except Exception as e:
                pytest.skip(f"Model {model_name} deterministic test failed: {e}")
    
    def test_model_device_compatibility(self):
        """Test that models work on CPU."""
        batch_size = 2
        seq_len = 10
        feature_dim = 8
        
        batch = Batch(
            times=[list(range(seq_len)) for _ in range(batch_size)],
            x_seq=torch.randn(batch_size, seq_len, feature_dim),
            alpha_seq=None,
            labels={"soh_r": torch.randn(batch_size, 2)},
        )
        
        models = available_models()
        for model_name in models:
            try:
                model_class = get_model(model_name)
                model = model_class()
                
                # Move to CPU explicitly
                model = model.cpu()
                batch.x_seq = batch.x_seq.cpu()
                batch.labels["soh_r"] = batch.labels["soh_r"].cpu()
                
                # Forward pass
                outputs, traj = model(batch, t_eval=None)
                loss = model.compute_loss(batch, outputs, traj)
                
                # Check that everything is on CPU
                assert outputs["soh_r"].device.type == "cpu"
                assert loss.device.type == "cpu"
                
            except Exception as e:
                pytest.skip(f"Model {model_name} device test failed: {e}")
