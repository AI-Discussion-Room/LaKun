"""One-command LaKun training, including automatic local multi-GPU DDP."""

import argparse
import subprocess
import sys

import torch

from lakun.train import main as training_main


def main(argv: list[str] | None = None) -> None:
    launcher = argparse.ArgumentParser(add_help=False)
    launcher.add_argument("--gpus", type=int, default=None,
                          help="number of visible GPUs to use (default: all visible GPUs)")
    launch_args, training_args = launcher.parse_known_args(argv)
    if launch_args.gpus is not None and launch_args.gpus < 1:
        launcher.error("--gpus must be positive")
    available_gpus = torch.cuda.device_count()
    gpu_count = launch_args.gpus if launch_args.gpus is not None else max(1, available_gpus)
    if launch_args.gpus is not None and available_gpus < gpu_count:
        launcher.error(f"requested {gpu_count} GPUs, found {available_gpus}")
    if gpu_count == 1:
        training_main(training_args)
    else:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node", str(gpu_count), "--module", "lakun.train",
                   *training_args]
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
