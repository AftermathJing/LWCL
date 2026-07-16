from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass


@dataclass
class GpuStat:
    index: int
    memory_used: int
    memory_total: int
    utilization: int

    @property
    def memory_ratio(self) -> float:
        return self.memory_used / self.memory_total if self.memory_total else 1.0


def load_stats() -> list[GpuStat]:
    command = [
        'nvidia-smi',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    output = subprocess.check_output(command, text=True)
    rows: list[GpuStat] = []
    reader = csv.reader(line for line in output.splitlines() if line.strip())
    for row in reader:
        rows.append(
            GpuStat(
                index=int(row[0].strip()),
                memory_used=int(row[1].strip()),
                memory_total=int(row[2].strip()),
                utilization=int(row[3].strip()),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Select the least loaded visible GPU')
    parser.add_argument('--max-used-mib', type=int, default=4096)
    parser.add_argument('--max-util', type=int, default=30)
    args = parser.parse_args()

    stats = load_stats()
    preferred = [
        gpu for gpu in stats
        if gpu.memory_used <= args.max_used_mib and gpu.utilization <= args.max_util
    ]
    candidates = preferred or stats
    candidates.sort(key=lambda gpu: (gpu.memory_used, gpu.utilization, gpu.index))
    print(candidates[0].index)


if __name__ == '__main__':
    main()
