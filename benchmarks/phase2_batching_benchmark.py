import argparse
import json
import time
import itertools
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from collision_backend import create_collision_backend
from placement import placementProcedure


@dataclass
class BatchBenchmarkResult:
    backend: str
    decodes: int
    warmup: int
    total_seconds: float
    ms_per_decode: float
    decodes_per_sec: float


def load_instance(nb_parts, nb_machines, inst_number):
    text = Path(f"data/Instances/P{nb_parts}M{nb_machines}-{inst_number}.txt").read_text()
    instance_parts = np.array([int(x) for x in text.split()])
    instance_parts_unique = np.unique(instance_parts)
    return instance_parts, instance_parts_unique


def build_parts_data(nb_machines, instance_parts_unique, job_spec, mach_spec, areas, backend):
    data = {}
    for m in range(nb_machines):
        data[m] = {}
        bin_length = int(mach_spec["L(mm)"].iloc[m])
        bin_width = int(mach_spec["W(mm)"].iloc[m])
        data[m]["binLength"] = bin_length
        data[m]["binWidth"] = bin_width
        data[m]["binArea"] = bin_length * bin_width
        data[m]["setupTime"] = float(mach_spec["ST(s)"].iloc[m])

        for part in instance_parts_unique:
            matrix = np.load(f"data/partsMatrices/matrix_{part}.npy").astype(np.int32)
            matrix = np.ascontiguousarray(matrix)
            nrot = 2 if np.array_equal(matrix, np.rot90(matrix, 2)) else 4

            data.setdefault(f"part{part}", {})
            data[m].setdefault(f"part{part}", {})
            for rot in range(nrot):
                rot_matrix = np.rot90(matrix, rot)
                data[f"part{part}"][f"rot{rot}"] = rot_matrix
                data[f"part{part}"][f"dens{rot}"] = np.array(
                    [max(len(list(g)) for k, g in itertools.groupby(row) if k) for row in rot_matrix]
                )
                data[f"part{part}"][f"shapes{rot}"] = [rot_matrix.shape[0], rot_matrix.shape[1]]
                data[m][f"part{part}"][f"fft{rot}"] = backend.prepare_part_fft(rot_matrix, bin_length, bin_width)

            data[f"part{part}"]["area"] = float(areas[part])
            data[m][f"part{part}"]["procTime"] = float(
                job_spec["volume(mm3)"].loc[part] * mach_spec["VT(s/mm3)"].iloc[m]
                + job_spec["support(mm3)"].loc[part] * mach_spec["SPT(s/mm3)"].iloc[m]
            )
            data[m][f"part{part}"]["procTimeHeight"] = float(
                job_spec["height(mm)"].loc[part] * mach_spec["HT(s/mm3)"].iloc[m]
            )
            data[f"part{part}"]["nrot"] = nrot
            data[f"part{part}"]["id"] = int(part)
            data[f"part{part}"]["lengths"] = [data[f"part{part}"][f"shapes{r}"][0] for r in range(nrot)]

    return data


def evaluate_batch(data, nb_parts, nb_machines, thresholds, instance_parts, backend, chromosomes, warmup):
    for i in range(warmup):
        placementProcedure(data, nb_parts, nb_machines, thresholds, chromosomes[i], instance_parts, backend)
    if "gpu" in backend.name:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for c in chromosomes[warmup:]:
        placementProcedure(data, nb_parts, nb_machines, thresholds, c, instance_parts, backend)
    if "gpu" in backend.name:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    measured = len(chromosomes) - warmup
    return BatchBenchmarkResult(
        backend=backend.name,
        decodes=measured,
        warmup=warmup,
        total_seconds=elapsed,
        ms_per_decode=(elapsed / measured) * 1000 if measured else 0.0,
        decodes_per_sec=measured / elapsed if elapsed > 0 else 0.0,
    )


def write_md(path, config, results):
    lines = []
    lines.append("# Phase 2 Batching Benchmark Report")
    lines.append("")
    lines.append("## Config")
    for k, v in config.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Results")
    for r in results:
        lines.append(
            f"- `{r.backend}`: ms_per_decode=`{r.ms_per_decode:.6f}`, decodes_per_sec=`{r.decodes_per_sec:.3f}`, total_seconds=`{r.total_seconds:.3f}`"
        )
    if len(results) == 2 and results[0].ms_per_decode > 0:
        speedup = results[0].ms_per_decode / results[1].ms_per_decode
        lines.append("")
        lines.append("## Speedup")
        lines.append(f"- Batched speedup vs unbatched (ms/decode): `{speedup:.3f}x`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Phase 2 batching in full decoder loop.")
    parser.add_argument("--nb-parts", type=int, default=50)
    parser.add_argument("--nb-machines", type=int, default=2)
    parser.add_argument("--inst-number", type=int, default=0)
    parser.add_argument("--decodes", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default="reports/phase2_batching_benchmark.json")
    parser.add_argument("--md-out", default="reports/PHASE2_BATCHING_REPORT.md")
    args = parser.parse_args()

    if args.warmup >= args.decodes:
        raise ValueError("--warmup must be smaller than --decodes")

    instance_parts, instance_parts_unique = load_instance(args.nb_parts, args.nb_machines, args.inst_number)
    job_spec_all = pd.read_excel("data/PartsMachines/part-machine-information.xlsx", sheet_name="part", header=0, index_col=0)
    job_spec = job_spec_all.loc[instance_parts_unique]
    mach_spec = pd.read_excel("data/PartsMachines/part-machine-information.xlsx", sheet_name="machine", header=0, index_col=0)
    areas = pd.read_excel("data/PartsMachines/polygon_areas.xlsx", header=0)["Area"].tolist()
    thresholds = [t / args.nb_machines for t in range(1, args.nb_machines)]

    rng = np.random.default_rng(args.seed)
    chromosomes = rng.uniform(0.0, 1.0, size=(args.decodes, 2 * args.nb_parts)).astype(np.float32)

    backends = [create_collision_backend("torch_gpu_unbatched"), create_collision_backend("torch_gpu")]
    results = []
    for backend in backends:
        data = build_parts_data(args.nb_machines, instance_parts_unique, job_spec, mach_spec, areas, backend)
        result = evaluate_batch(
            data,
            args.nb_parts,
            args.nb_machines,
            thresholds,
            instance_parts,
            backend,
            chromosomes,
            args.warmup,
        )
        results.append(result)

    payload = {
        "config": vars(args),
        "results": [asdict(r) for r in results],
        "speedup_unbatched_to_batched": results[0].ms_per_decode / results[1].ms_per_decode if results[1].ms_per_decode > 0 else None,
    }

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_md(md_path, vars(args), results)

    print(json.dumps(payload, indent=2))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
