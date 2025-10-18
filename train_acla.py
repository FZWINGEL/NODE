#!/usr/bin/env python3
"""
Training script for ACLA model on NASA battery dataset.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add mlbench to path
sys.path.insert(0, str(Path(__file__).parent / "mlbench" / "src"))

from mlbench.data.registry import get_dataset
from mlbench.models import ACLAModel
from mlbench.train.run import train_model
from mlbench.eval.run import evaluate_model
from mlbench.utils.mlflow_utils import setup_mlflow


def main():
    parser = argparse.ArgumentParser(description="Train ACLA model")
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
    parser.add_argument("--v_min", type=float, default=3.6, 
                       help="Minimum voltage for sampling")
    parser.add_argument("--v_max", type=float, default=4.2, 
                       help="Maximum voltage for sampling")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--save_dir", type=str, default="artifacts", 
                       help="Directory to save model artifacts")
    parser.add_argument("--experiment_name", type=str, default="acla_nasa", 
                       help="MLflow experiment name")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup MLflow
    setup_mlflow(experiment_name=args.experiment_name)
    
    # Load dataset
    print("Loading NASA dataset...")
    train_loader, val_loader = get_dataset(
        "nasa_cc",
        batch_size=args.batch_size,
        val_split=0.2,
        seed=42,
        data_dir=args.data_dir,
        window=args.window,
        stride=args.stride,
        nv=args.nv,
        v_min=args.v_min,
        v_max=args.v_max
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating ACLA model...")
    model = ACLAModel(
        input_dim=args.nv + 1,  # SOH + voltage time points
        attention_dim=10,
        attention_width=5,
        cnn_hidden_dim=64,
        lstm_hidden_dim=64,
        state_dim=64,
        aug_dim=20,
        output_dim=2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer (AdamW + Lookahead as per paper)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler (three-phase protocol from paper)
    def lr_scheduler(epoch):
        if epoch < 220:  # Warm-up phase
            return epoch / 220
        elif epoch < 720:  # Plateau phase (220 + 500)
            return 1.0
        else:  # Decay phase
            decay_epochs = args.epochs - 720
            return max(1e-5 / args.lr, 1.0 - (epoch - 720) / decay_epochs)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.__dict__.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, traj = model(batch)
            loss = model.compute_loss(batch, outputs, traj)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.__dict__.items()}
                
                outputs, traj = model(batch)
                loss = model.compute_loss(batch, outputs, traj)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'acla_best.pt'))
            print(f"Saved best model at epoch {epoch}")
        
        # Early stopping
        if epoch > 50 and val_loss > best_val_loss * 1.1:
            print("Early stopping triggered")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, os.path.join(args.save_dir, 'acla_final.pt'))
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Evaluation
    print("Evaluating model...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'acla_best.pt'))['model_state_dict'])
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.__dict__.items()}
            
            outputs, traj = model(batch)
            predictions.extend(outputs['soh_r'][:, 0].cpu().numpy())
            targets.extend(batch.labels['soh_r'][:, 0].cpu().numpy())
    
    # Calculate metrics
    predictions = torch.tensor(predictions)
    targets = torch.tensor(targets)
    
    mse = torch.mean((predictions - targets) ** 2)
    mae = torch.mean(torch.abs(predictions - targets))
    rmse = torch.sqrt(mse)
    mape = torch.mean(torch.abs((predictions - targets) / (targets + 1e-8))) * 100
    
    print(f"Final Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()
