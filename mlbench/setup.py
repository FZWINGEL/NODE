from setuptools import setup, find_packages
import warnings

# Deprecation warning for setup.py
warnings.warn(
    "setup.py is deprecated. Use 'pip install -e .' with pyproject.toml instead.",
    DeprecationWarning,
    stacklevel=2,
)

setup(
    name="mlbench",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0",
        "torchdiffeq>=0.2.3",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.4.0",
        "mlflow>=2.14.0",
        "optuna>=3.6.0",
        "ax-platform>=0.3.6",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
    ],
    entry_points={
        "console_scripts": [
            # New CLI commands
            "soh-train=mlbench.experiments.run:main",
            "soh-eval=mlbench.experiments.eval:main",
            "soh-sweep=mlbench.experiments.sweep:main",
            "soh-hpo=mlbench.experiments.hpo:main",
            # Legacy commands (deprecated)
            "mlbench-train=mlbench.train.run:main",
            "mlbench-eval=mlbench.eval.run:main",
        ],
    },
)
