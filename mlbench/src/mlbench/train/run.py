import argparse
import os
import torch
import mlflow
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..utils.registry import get as get_model
from ..utils.mlflow_utils import start_run
from ..utils.seed import set_seed
from ..data.registry import get_dataset
# Import models to register them
import mlbench.models


def train_one_epoch(model, loader, opt, device):
	model.train()
	total = 0.0
	for batch in loader:
		# Move batch data to GPU
		batch.x_seq = batch.x_seq.to(device)
		batch.labels["soh_r"] = batch.labels["soh_r"].to(device)
		outputs, traj = model(batch, t_eval=None)
		loss = model.compute_loss(batch, outputs, traj)
		opt.zero_grad()
		loss.backward()
		# Clip gradients to prevent explosion
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		opt.step()
		total += float(loss.item())
	return total / max(1, len(loader))


def evaluate(model, loader, device):
	model.eval()
	mse = 0.0
	with torch.no_grad():
		for batch in loader:
			# Move batch data to GPU
			batch.x_seq = batch.x_seq.to(device)
			batch.labels["soh_r"] = batch.labels["soh_r"].to(device)
			outputs, traj = model(batch, t_eval=None)
			loss = model.compute_loss(batch, outputs, traj)
			mse += float(loss.item())
	return {"val_mse": mse / max(1, len(loader))}


def run(
	model: str = "lstm",
	epochs: int = 10,
	lr: float = 1e-3,
	batch_size: int = 32,
	seed: int = 0,
	data_name: str = "dummy",
	data_dir: str | None = None,
	window: int = 20,
	stride: int = 5,
	val_split: float = 0.2,
	model_kwargs: dict | None = None,
	nested_run: bool = False,
	tags: dict | None = None,
) -> float:
	# Set global seeds for reproducibility
	set_seed(seed)
	Model = get_model(model)
	m = Model(**(model_kwargs or {}))
	# Ensure model is on GPU if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	m = m.to(device)
	print(f"Using device: {device}")
	opt = AdamW(m.parameters(), lr=lr)
	sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
	builder = get_dataset(data_name)
	builder_kwargs = {"batch_size": batch_size, "val_split": val_split, "seed": seed}
	if data_name == "nasa":
		builder_kwargs.update({"data_dir": data_dir, "window": window, "stride": stride})
	elif data_name == "nasa_cc":
		builder_kwargs.update({"data_dir": data_dir, "window": window, "stride": stride})
	train_loader, val_loader = builder(**builder_kwargs)
	with start_run(run_name=f"train_{model}", tags=tags, nested=nested_run):
		mlflow.log_params({
			"model": model,
			"epochs": epochs,
			"lr": lr,
			"batch_size": batch_size,
			"seed": seed,
			"data_name": data_name,
			"data_dir": data_dir or "<default>",
			"window": window,
			"stride": stride,
			"val_split": val_split,
		})
		# Log model constructor kwargs (flattened)
		if model_kwargs:
			flat_params = {f"model.{k}": v for k, v in model_kwargs.items()}
			mlflow.log_params(flat_params)
		best_val = float("inf")
		for ep in range(epochs):
			train_loss = train_one_epoch(m, train_loader, opt, device)
			val_metrics = evaluate(m, val_loader, device)
			sched.step(val_metrics["val_mse"])
			mlflow.log_metrics({"train_loss": train_loss, **val_metrics}, step=ep)
			print({"epoch": ep, "train_loss": train_loss, **val_metrics})
			if val_metrics["val_mse"] < best_val:
				best_val = val_metrics["val_mse"]
		# save model
		os.makedirs("artifacts", exist_ok=True)
		ckpt = os.path.join("artifacts", f"{model}.pt")
		torch.save(m.state_dict(), ckpt)
		mlflow.log_artifact(ckpt)
		mlflow.log_metric("best_val_mse", best_val)
	return best_val

def main():
	p = argparse.ArgumentParser()
	p.add_argument("--model", type=str, default="lstm")
	p.add_argument("--epochs", type=int, default=10)
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--batch_size", type=int, default=32)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--data_name", type=str, default="dummy")
	p.add_argument("--data_dir", type=str, default=None)
	p.add_argument("--window", type=int, default=20)
	p.add_argument("--stride", type=int, default=5)
	p.add_argument("--val_split", type=float, default=0.2)
	# Model-specific parameters
	p.add_argument("--input_dim", type=int, default=None)
	p.add_argument("--hidden_dim", type=int, default=None)
	p.add_argument("--num_layers", type=int, default=None)
	p.add_argument("--output_dim", type=int, default=None)
	args = p.parse_args()
	
	# Build model kwargs from provided arguments
	model_kwargs = {}
	if args.input_dim is not None:
		model_kwargs["input_dim"] = args.input_dim
	if args.hidden_dim is not None:
		model_kwargs["hidden_dim"] = args.hidden_dim
	if args.num_layers is not None:
		model_kwargs["num_layers"] = args.num_layers
	if args.output_dim is not None:
		model_kwargs["output_dim"] = args.output_dim
	
	run(
		model=args.model,
		epochs=args.epochs,
		lr=args.lr,
		batch_size=args.batch_size,
		seed=args.seed,
		data_name=args.data_name,
		data_dir=args.data_dir,
		window=args.window,
		stride=args.stride,
		val_split=args.val_split,
		model_kwargs=model_kwargs if model_kwargs else None,
	)

if __name__ == "__main__":
	main()
