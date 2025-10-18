from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from ...utils.registry import register
from ..base import ForwardModel


class AttentionLayer(nn.Module):
    """Attention mechanism for feature weighting as described in ACLA paper."""
    
    def __init__(self, input_dim: int, attention_dim: int, attention_width: int):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.attention_width = attention_width
        
        # Linear transformation for attention scores
        self.attention_linear = nn.Linear(input_dim, attention_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Attended features of shape [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Calculate attention scores
        attention_scores = self.attention_linear(x)  # [B, T, attention_dim]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [B, T, attention_dim]
        
        # Apply attention to consecutive features
        # For simplicity, we'll apply attention to the last attention_width features
        if self.attention_width > 0 and self.input_dim > self.attention_width:
            start_idx = self.input_dim - self.attention_width
            attended_features = x[:, :, start_idx:] * attention_weights[:, :, :self.attention_width]
            # Concatenate with original features
            x_attended = torch.cat([x[:, :, :start_idx], attended_features], dim=-1)
        else:
            x_attended = x
            
        return x_attended


class CNNLSTMEncoder(nn.Module):
    """CNN-LSTM encoder as described in ACLA paper."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Two 1D CNN layers: 64 and 32 filters
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        
        # LSTM layer with 64 hidden units
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Encoded features of shape [batch_size, hidden_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Transpose for Conv1d: [B, D, T]
        x_conv = x.transpose(1, 2)
        
        # Apply CNN layers
        x_conv = torch.relu(self.conv1(x_conv))
        x_conv = torch.relu(self.conv2(x_conv))
        
        # Transpose back: [B, T, D]
        x_conv = x_conv.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x_conv)
        
        # Take the last output
        return lstm_out[:, -1, :]  # [B, hidden_dim]


class ANODEDynamics(nn.Module):
    """ANODE dynamics function with augmented dimensions."""
    
    def __init__(self, state_dim: int, aug_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.aug_dim = aug_dim
        self.total_dim = state_dim + aug_dim
        
        # MLP for dynamics with more stable initialization
        self.dynamics_net = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.Tanh(),  # Use Tanh instead of ReLU for stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.total_dim)
        )
        
        # Initialize weights for stability
        for layer in self.dynamics_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
        
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor
            y: State tensor of shape [batch_size, state_dim + aug_dim]
        Returns:
            Derivative dy/dt
        """
        # Add small epsilon to prevent numerical issues
        y_safe = y + 1e-8
        output = self.dynamics_net(y_safe)
        # Clip gradients to prevent explosion
        return torch.clamp(output, -10.0, 10.0)


@register("acla")
class ACLAModel(ForwardModel):
    """
    ACLA (Attention-based CNN-LSTM-ANODE) model for battery SOH prediction.
    
    Architecture:
    1. Attention mechanism for feature weighting
    2. Two 1D CNN layers (64, 32 filters)
    3. LSTM layer (64 hidden units)
    4. ANODE solver with augmented dimensions
    5. Linear output layer
    """
    
    def __init__(
        self, 
        input_dim: int = 20,  # SOH + 19 voltage time points (NASA dataset)
        attention_dim: int = 10,
        attention_width: int = 5,
        cnn_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        state_dim: int = 64,
        aug_dim: int = 20,  # As specified in paper
        output_dim: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.aug_dim = aug_dim
        self.output_dim = output_dim
        
        # Attention layer
        self.attention = AttentionLayer(input_dim, attention_dim, attention_width)
        
        # CNN-LSTM encoder
        self.encoder = CNNLSTMEncoder(input_dim, lstm_hidden_dim)
        
        # State projection
        self.state_proj = nn.Linear(lstm_hidden_dim, state_dim)
        
        # ANODE dynamics
        self.dynamics = ANODEDynamics(state_dim, aug_dim)
        
        # Output head
        self.head = nn.Linear(state_dim, output_dim)
        
    def forward(self, batch, t_eval: torch.Tensor | None = None):
        """
        Forward pass of ACLA model.
        
        Args:
            batch: Input batch containing x_seq
            t_eval: Evaluation times (not used in this implementation)
        """
        x = batch.x_seq  # [B, T, D]
        
        # Apply attention mechanism
        x_attended = self.attention(x)
        
        # Encode with CNN-LSTM
        encoded = self.encoder(x_attended)  # [B, lstm_hidden_dim]
        
        # Project to state dimension
        y0 = self.state_proj(encoded)  # [B, state_dim]
        
        # Initialize augmented dimensions
        batch_size = y0.size(0)
        a0 = torch.zeros(batch_size, self.aug_dim, device=y0.device, dtype=y0.dtype)
        
        # Concatenate state and augmented dimensions
        z0 = torch.cat([y0, a0], dim=-1)  # [B, state_dim + aug_dim]
        
        # Define time points for ODE integration
        t = torch.tensor([0.0, 1.0], device=y0.device)
        
        # Solve ODE using ANODE
        zt = odeint(
            self.dynamics, 
            z0, 
            t, 
            method='dopri5', 
            rtol=1e-5, 
            atol=1e-6
        )
        
        # Extract final state (without augmented dimensions)
        z_final = zt[-1]  # [B, state_dim + aug_dim]
        y_final = z_final[:, :self.state_dim]  # [B, state_dim]
        
        # Generate prediction
        pred = self.head(y_final)  # [B, output_dim]
        
        return {"soh_r": pred}, {"state_traj": zt}
    
    def compute_loss(self, batch, outputs, traj):
        """Compute MSE loss for SOH prediction."""
        return torch.nn.functional.mse_loss(outputs["soh_r"], batch.labels["soh_r"])
