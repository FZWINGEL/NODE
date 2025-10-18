import argparse
import os
import json
import torch
from .metrics import rmse
from ..utils.registry import get as get_model
from ..data.registry import get_dataset
# Import models to register them
import mlbench.models


def run(model: str = "lstm", checkpoint: str | None = None, batch_size: int = 32, data_name: str = "dummy", data_dir: str | None = None, window: int = 20, stride: int = 5, val_split: float = 0.2):
	Model = get_model(model)
	m = Model()
	if checkpoint and os.path.isfile(checkpoint):
		m.load_state_dict(torch.load(checkpoint, map_location="cpu"))
	builder = get_dataset(data_name)
	builder_kwargs = {"batch_size": batch_size, "val_split": val_split, "seed": 0}
	if data_name == "nasa":
		builder_kwargs.update({"data_dir": data_dir, "window": window, "stride": stride})
	elif data_name == "nasa_cc":
		builder_kwargs.update({"data_dir": data_dir, "window": window, "stride": stride})
	_, val_loader = builder(**builder_kwargs)
	m.eval()
	outs = []
	with torch.no_grad():
		for batch in val_loader:
			pred, _ = m(batch, t_eval=None)
			outs.append((pred["soh_r"], batch.labels["soh_r"]))
	preds = torch.cat([p for p, t in outs], dim=0)
	targs = torch.cat([t for p, t in outs], dim=0)
	return {"rmse": rmse(preds, targs)}

def main():
	p = argparse.ArgumentParser()
	p.add_argument("--model", type=str, default="lstm")
	p.add_argument("--checkpoint", type=str, default=None)
	p.add_argument("--batch_size", type=int, default=32)
	p.add_argument("--data_name", type=str, default="dummy")
	p.add_argument("--data_dir", type=str, default=None)
	p.add_argument("--window", type=int, default=20)
	p.add_argument("--stride", type=int, default=5)
	p.add_argument("--val_split", type=float, default=0.2)
	args = p.parse_args()
	print(json.dumps(run(model=args.model, checkpoint=args.checkpoint, batch_size=args.batch_size, data_name=args.data_name, data_dir=args.data_dir, window=args.window, stride=args.stride, val_split=args.val_split)))

if __name__ == "__main__":
	main()
