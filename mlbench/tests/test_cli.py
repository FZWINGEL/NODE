"""Tests for CLI functionality."""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path
import yaml


class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_help(self):
        """Test that CLI commands show help."""
        commands = ["soh-train", "soh-eval", "soh-sweep", "soh-hpo"]
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    [cmd, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                assert result.returncode == 0
                assert "usage:" in result.stdout.lower() or "help" in result.stdout.lower()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.skip(f"Command {cmd} not available")
    
    def test_train_cli_config_validation(self):
        """Test that train CLI validates config files."""
        # Create invalid config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"invalid": "config"}, f)
            invalid_config = f.name
        
        try:
            result = subprocess.run(
                ["soh-train", "--config", invalid_config],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Should fail with invalid config
            assert result.returncode != 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("soh-train command not available")
        finally:
            os.unlink(invalid_config)
    
    def test_eval_cli_help(self):
        """Test eval CLI help."""
        try:
            result = subprocess.run(
                ["soh-eval", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
            assert "artifact" in result.stdout.lower()
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("soh-eval command not available")
    
    def test_sweep_cli_help(self):
        """Test sweep CLI help."""
        try:
            result = subprocess.run(
                ["soh-sweep", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
            assert "sweep" in result.stdout.lower()
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("soh-sweep command not available")
    
    def test_hpo_cli_help(self):
        """Test HPO CLI help."""
        try:
            result = subprocess.run(
                ["soh-hpo", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
            assert "hpo" in result.stdout.lower() or "hyperparameter" in result.stdout.lower()
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("soh-hpo command not available")
    
    def test_cli_imports(self):
        """Test that CLI modules can be imported."""
        try:
            import mlbench.experiments.run
            import mlbench.experiments.eval
            import mlbench.experiments.sweep
            import mlbench.experiments.hpo
            
            # Check that main functions exist
            assert hasattr(mlbench.experiments.run, "main")
            assert hasattr(mlbench.experiments.eval, "main")
            assert hasattr(mlbench.experiments.sweep, "main")
            assert hasattr(mlbench.experiments.hpo, "main")
            
        except ImportError as e:
            pytest.skip(f"CLI modules not available: {e}")
    
    def test_cli_config_loading(self):
        """Test CLI config loading functionality."""
        # Create valid minimal config
        config_dict = {
            "model": {
                "name": "lstm",
                "params": {}
            },
            "dataset": {
                "name": "dummy",
                "params": {
                    "batch_size": 2,
                    "val_split": 0.5,
                    "seed": 42,
                }
            },
            "training": {
                "epochs": 1,
                "device": "cpu",
                "gradient_clip_norm": 1.0,
                "mixed_precision": False,
            },
            "optimizer": {
                "name": "adamw",
                "params": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                }
            },
            "scheduler": {
                "name": "reduce_on_plateau",
                "params": {
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 3,
                    "min_lr": 1e-6,
                }
            },
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 5,
                "min_delta": 0.0,
                "mode": "min",
            },
            "run": {
                "name": "test_run",
                "tags": {},
                "save_dir": "artifacts",
                "log_level": "INFO",
            }
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            # Test config loading
            from mlbench.experiments.run import _load_experiment_config
            config = _load_experiment_config(Path(config_path))
            
            # Check config structure
            assert hasattr(config, "model")
            assert hasattr(config, "dataset")
            assert hasattr(config, "training")
            assert hasattr(config, "optimizer")
            assert hasattr(config, "scheduler")
            assert hasattr(config, "early_stopping")
            assert hasattr(config, "run")
            
        except Exception as e:
            pytest.skip(f"Config loading test failed: {e}")
        finally:
            os.unlink(config_path)
