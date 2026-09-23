"""Two-process CPU rehearsal of the same DDP path used by the GPU launcher."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from lakun.inference import LaKunPredictor


def test_two_rank_training(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    output = tmp_path / "ddp-checkpoint"
    command = [sys.executable, "-m", "lakun.train", "--smoke", "--device", "cpu",
               "--train-groups", "4", "--val-groups", "4", "--test-groups", "4", "--epochs", "1", "--out", str(output),
               "--log-every", "0"]
    base_env = os.environ.copy()
    base_env.update({"WORLD_SIZE": "2", "MASTER_ADDR": "127.0.0.1",
                     "MASTER_PORT": str(port), "OMP_NUM_THREADS": "1"})
    processes = []
    try:
        for rank in range(2):
            environment = {**base_env, "RANK": str(rank), "LOCAL_RANK": str(rank)}
            processes.append(subprocess.Popen(command, cwd=root, env=environment,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              text=True, encoding="utf-8", errors="replace",
                                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        results = [process.communicate(timeout=120) for process in processes]
        for process, (stdout, _) in zip(processes, results):
            assert process.returncode == 0, stdout[-3000:]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["distributed_world_size"] == 2
    assert report["history"][0]["train"]["groups"] == 4
    assert report["validation_checks"][0]["val"]["groups"] == 4
    assert report["test"]["groups"] == 4
    assert report["best_step"] == 2
    assert (output / "best" / "lakun.safetensors").is_file()
    assert LaKunPredictor.from_pretrained(output / "best", device="cpu").model_status == "tiny_random_smoke"
