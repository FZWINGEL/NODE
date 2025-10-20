#!/usr/bin/env python3
"""
Training script for ACLA model using mlbench infrastructure.
"""

import argparse
import sys
from pathlib import Path

# Add mlbench to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlbench.train.run import run


def main():
    parser = argparse.ArgumentParser(description="Train ACLA model using mlbench")
    parser.add_argument("--data_dir", type=str, default="mlbench/data/NASA", 
                       help="Path to NASA dataset")
    parser.add_argument("--epochs", type=int, default=1000, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Batch size (paper uses 1)")
    parser.add_argument("--lr", type=float, default=0.01, 
                       help="Learning rate")
    parser.add_argument("--window", type=int, default=20, 
                       help="Sequence window size")
    parser.add_argument("--stride", type=int, default=5, 
                       help="Sequence stride")
    parser.add_argument("--nv", type=int, default=19, 
                       help="Number of voltage sampling points")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--save_dir", type=str, default="artifacts", 
                       help="Directory to save model artifacts")
    
    args = parser.parse_args()
    
    # ACLA model configuration
    model_kwargs = {
        "input_dim": args.nv + 1,  # SOH + voltage time points
        "attention_dim": 10,
        "attention_width": 5,
        "cnn_hidden_dim": 64,
        "lstm_hidden_dim": 64,
        "state_dim": 64,
        "aug_dim": 20,
        "output_dim": 2
    }
    
    print("Starting ACLA model training using mlbench infrastructure...")
    print(f"Model configuration: {model_kwargs}")
    
    # Run training using mlbench infrastructure
    best_val_loss = run(
        model="acla",
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=42,
        data_name="nasa_cc",
        data_dir=args.data_dir,
        window=args.window,
        stride=args.stride,
        val_split=0.2,
        model_kwargs=model_kwargs,
        nested_run=False,
        tags={"model": "acla", "dataset": "nasa"}
    )
    
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
