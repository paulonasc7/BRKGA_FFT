import argparse
import itertools
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from collision_backend import create_collision_backend
from placement import placementProcedure
from concurrent.futures import ThreadPoolExecutor


@dataclass
class TuneResult:
    workers: int
    chunksize: int
    repeat: int
    ms_per_decode: float
    decodes_per_sec: float
    total_seconds: float


def load_problem(nb_parts, nb_machines, inst_number):
    text = Path(f"data/Instances/P{nb_parts}M{nb_machines}-{inst_number}.txt").read_text()
    instance_parts = np.array([int(x) for x in text.split()])
    unique_parts = np.unique(instance_parts)
    return instance_parts, unique_parts


def build_data(nb_machines, unique_parts, job_spec, mach_spec, area, backend):
    data = {}
    for m in range(nb_machines):
        data[m] = {}
        bin_length = int(mach_spec["L(mm)"].iloc[m])
        bin_width = int(mach_spec["W(mm)"].iloc[m])
        data[m]["binLength"] = bin_length
        data[m]["binWidth"] = bin_width
        data[m]["binArea"] = bin_length * bin_width
        data[m]["setupTime"] = float(mach_spec["ST(s)"].iloc[m])

        for part in unique_parts:
            matrix = np.load(f"data/partsMatrices/matrix_{part}.npy").astype(np.int32)
            matrix = np.ascontiguousarray(matrix)
            nrot = 2 if np.array_equal(matrix, np.rot90(matrix, 2)) else 4

            data.setdefault(f"part{part}", {})
            data[m].setdefault(f"part{part}", {})
            for rot in range(nrot):
                rot_matrix = np.ascontiguousarray(np.rot90(matrix, rot))
                data[f"part{part}"][f"rot{rot}"] = rot_matrix
                data[f"part{part}"][f"dens{rot}"] = np.array(
                    [max(len(list(g)) for k, g in itertools.groupby(row) if k) for row in rot_matrix]
                )
                data[f"part{part}"][f"shapes{rot}"] = [rot_matrix.shape[0], rot_matrix.shape[1]]
                data[m][f"part{part}"][f"fft{rot}"] = backend.prepare_part_fft(rot_matrix, bin_length, bin_width)

            data[f"part{part}"]["area"] = float(area[part])
            data[m][f"part{part}"]["procTime"] = float(
                job_spec["volume(mm3)"].loc[part] * mach_spec["VT(s/mm3)"].iloc[m]
                + job_spec["support(mm3)"].loc[part] * mach_spec["SPT(s/mm3)"].iloc[m]
            )
            data[m][f"part{part}"]["procTimeHeight"] = float(
                job_spec["height(mm)"].loc[part] * mach_spec["HT(s/mm3)"].iloc[m]
            )
            data[f"part{part}"]["nrot"] = nrot
            data[f"part{part}"]["id"] = int(part)
            data[f"part{part}"]["lengths"] = [data[f"part{part}"][f"shapes{i}"][0] for i in range(nrot)]
    return data


def evaluate_solution(args):
    parts_dict, nb_parts, nb_machines, thresholds, chromosome, instance_parts, backend = args
    return placementProcedure(parts_dict, nb_parts, nb_machines, thresholds, chromosome, instance_parts, backend)


def run_config(parts_dict, nb_parts, nb_machines, thresholds, instance_parts, backend, chromosomes, warmup, workers, chunksize):
    tasks = [(parts_dict, nb_parts, nb_machines, thresholds, c, instance_parts, backend) for c in chromosomes]

    if warmup > 0:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(evaluate_solution, tasks[:warmup], chunksize=chunksize))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(evaluate_solution, tasks[warmup:], chunksize=chunksize))
    elapsed = time.perf_counter() - start
    measured = len(chromosomes) - warmup
    return (elapsed / measured) * 1000, measured / elapsed, elapsed


def parse_ints(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def write_md(path, config, results):
    grouped = {}
    for r in results:
        key = (r.workers, r.chunksize)
        grouped.setdefault(key, []).append(r.ms_per_decode)

    summary = []
    for (workers, chunksize), vals in grouped.items():
        summary.append(
            {
                "workers": workers,
                "chunksize": chunksize,
                "median_ms": statistics.median(vals),
                "mean_ms": statistics.mean(vals),
                "runs": vals,
            }
        )
    summary.sort(key=lambda x: x["median_ms"])

    lines = ["# Phase 5 Parallel Tuning Report", "", "## Config"]
    for k, v in config.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Ranked Results (by median ms/decode)"]
    for i, s in enumerate(summary, start=1):
        lines.append(
            f"{i}. workers=`{s['workers']}` chunksize=`{s['chunksize']}` median_ms=`{s['median_ms']:.6f}` mean_ms=`{s['mean_ms']:.6f}` runs=`{[round(x, 6) for x in s['runs']]}`"
        )
    if summary:
        best = summary[0]
        lines += ["", "## Recommended Config", f"- workers=`{best['workers']}` chunksize=`{best['chunksize']}`"]

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 thread parallel tuning benchmark.")
    parser.add_argument("--nb-parts", type=int, default=100)
    parser.add_argument("--nb-machines", type=int, default=4)
    parser.add_argument("--inst-number", type=int, default=0)
    parser.add_argument("--backend", default="torch_gpu")
    parser.add_argument("--decodes", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--chunksizes", default="1,2,4,8")
    parser.add_argument("--json-out", default="reports/phase5_parallel_tuning.json")
    parser.add_argument("--md-out", default="reports/PHASE5_PARALLEL_TUNING.md")
    args = parser.parse_args()

    if args.warmup >= args.decodes:
        raise ValueError("--warmup must be smaller than --decodes")

    workers_grid = parse_ints(args.workers)
    chunks_grid = parse_ints(args.chunksizes)

    backend = create_collision_backend(args.backend)
    instance_parts, unique_parts = load_problem(args.nb_parts, args.nb_machines, args.inst_number)

    job_spec_all = pd.read_excel("data/PartsMachines/part-machine-information.xlsx", sheet_name="part", header=0, index_col=0)
    job_spec = job_spec_all.loc[unique_parts]
    mach_spec = pd.read_excel("data/PartsMachines/part-machine-information.xlsx", sheet_name="machine", header=0, index_col=0)
    area = pd.read_excel("data/PartsMachines/polygon_areas.xlsx", header=0)["Area"].tolist()
    thresholds = [t / args.nb_machines for t in range(1, args.nb_machines)]
    parts_dict = build_data(args.nb_machines, unique_parts, job_spec, mach_spec, area, backend)

    results = []
    for workers in workers_grid:
        for chunksize in chunks_grid:
            for repeat in range(args.repeats):
                rng = np.random.default_rng(args.seed + repeat)
                chromosomes = rng.uniform(0.0, 1.0, size=(args.decodes, 2 * args.nb_parts)).astype(np.float32)
                ms, dps, elapsed = run_config(
                    parts_dict, args.nb_parts, args.nb_machines, thresholds, instance_parts, backend, chromosomes,
                    args.warmup, workers, chunksize
                )
                results.append(TuneResult(workers, chunksize, repeat, ms, dps, elapsed))

    grouped = {}
    for r in results:
        key = (r.workers, r.chunksize)
        grouped.setdefault(key, []).append(r.ms_per_decode)
    ranking = []
    for (workers, chunksize), vals in grouped.items():
        ranking.append(
            {
                "workers": workers,
                "chunksize": chunksize,
                "median_ms_per_decode": statistics.median(vals),
                "mean_ms_per_decode": statistics.mean(vals),
                "all_ms_per_decode": vals,
            }
        )
    ranking.sort(key=lambda x: x["median_ms_per_decode"])

    payload = {
        "config": vars(args),
        "results": [asdict(r) for r in results],
        "ranking": ranking,
        "recommended": ranking[0] if ranking else None,
    }

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_md(md_path, vars(args), results)

    print(json.dumps(payload["recommended"], indent=2))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()

