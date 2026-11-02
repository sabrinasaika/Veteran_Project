#!/usr/bin/env python3
"""Hold a block of GPU memory for the experiment so other jobs cannot take it between model switches.

    python veteran_eval_memory.py <control file> <status file>

Write a number of GB to the control file and the holder resizes its reservation to that amount
(0 releases everything). The status file reports how many GB are currently held. The holder
releases all memory when it is stopped.
"""
from __future__ import annotations

import os
import signal
import sys
import time

import torch

CHUNK = 1024**3  # reserve in 1 GB pieces so the hold can shrink and grow precisely


def main() -> int:
    control, status = sys.argv[1], sys.argv[2]
    chunks: list[torch.Tensor] = []

    def resize(target_gb: int) -> None:
        while len(chunks) > target_gb:
            chunks.pop()
        torch.cuda.empty_cache()
        while len(chunks) < target_gb:
            try:
                chunks.append(torch.empty(CHUNK, dtype=torch.uint8, device="cuda"))
            except torch.cuda.OutOfMemoryError:
                break
        with open(status, "w", encoding="utf-8") as handle:
            handle.write(str(len(chunks)))

    def stop(*_: object) -> None:
        resize(0)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    resize(0)
    last = None
    while True:
        try:
            stamp = os.stat(control).st_mtime_ns
            if stamp != last:
                last = stamp
                with open(control, encoding="utf-8") as handle:
                    resize(max(0, int(float(handle.read().strip() or 0))))
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
