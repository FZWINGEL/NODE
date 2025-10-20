#!/usr/bin/env python3
"""
Test script for ACLA model to verify it works correctly.
"""

import sys
import torch
import pytest
from pathlib import Path

# Add mlbench to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlbench.models import ACLAModel
from mlbench.data.types import Batch


def test_acla_model():
    """Test ACLA model with dummy data."""
    print("Testing ACLA model...")
    
    # Create model
    model = ACLAModel(
        input_dim=20,  # SOH + 19 voltage time points
        attention_dim=10,
        attention_width=5,
        cnn_hidden_dim=64,
        lstm_hidden_dim=64,
        state_dim=64,
        aug_dim=20,
        output_dim=2
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy batch
    batch_size = 2
    seq_len = 20
    input_dim = 20
    
    # Create dummy data
    x_seq = torch.randn(batch_size, seq_len, input_dim)
    labels = torch.randn(batch_size, 2)  # SOH prediction + dummy
    
    batch = Batch(
        times=[list(range(seq_len)) for _ in range(batch_size)],
        x_seq=x_seq,
        alpha_seq=None,
        labels={"soh_r": labels}
    )
    
    print(f"Batch shape: {batch.x_seq.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs, traj = model(batch, t_eval=None)
    
    print(f"Output shape: {outputs['soh_r'].shape}")
    print(f"Trajectory shape: {traj['state_traj'].shape}")
    
    # Test loss computation
    loss = model.compute_loss(batch, outputs, traj)
    print(f"Loss: {loss.item():.6f}")
    
    # Test gradient computation
    model.train()
    outputs, traj = model(batch, t_eval=None)
    loss = model.compute_loss(batch, outputs, traj)
    loss.backward()
    
    print("Gradient computation successful!")
    
    # Check if gradients are computed
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"Parameters have gradients: {has_gradients}")
    
    print("ACLA model test completed successfully!")


if __name__ == "__main__":
    test_acla_model()
