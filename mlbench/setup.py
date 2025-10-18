from setuptools import setup, find_packages

setup(
    name="mlbench",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pandas",
        "mlflow",
        "optuna",
        "ax-platform",
        "botorch",
        "hydra-core",
        "omegaconf",
        "tqdm",
        "wandb",
        "torchdiffeq",
    ],
    entry_points={
        "console_scripts": [
            "mlbench-train=mlbench.train.run:main",
            "mlbench-eval=mlbench.eval.run:main",
        ],
    },
)
