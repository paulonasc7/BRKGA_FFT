import argparse
import json
import statistics
import time
import itertools
import sys
from dataclasses import dataclass, asdict
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
class RunResult:
    backend: str
    tf32: bool
    cufft_plan_cache: int
    repeat: int
    ms_per_decode: float
    decodes_per_sec: float
    total_seconds: float


def load_instance(nb_parts, nb_machines, inst_number):
    text = Path(f"data/Instances/P{nb_parts}M{nb_machines}-{inst_number}.txt").read_text()
    instance_parts = np.array([int(x) for x in text.split()])
    return instance_parts, np.unique(instance_parts)


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
                rot_matrix = np.ascontiguousarray(np.rot90(matrix, rot))
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


def evaluate_decodes(data, nb_parts, nb_machines, thresholds, instance_parts, backend, chromosomes, warmup):
    for i in range(warmup):
        placementProcedure(data, nb_parts, nb_machines, thresholds, chromosomes[i], instance_parts, backend)
    if "cuda" in backend.name:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for c in chromosomes[warmup:]:
        placementProcedure(data, nb_parts, nb_machines, thresholds, c, instance_parts, backend)
    if "cuda" in backend.name:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    measured = len(chromosomes) - warmup
    ms_per_decode = (elapsed / measured) * 1000
    return ms_per_decode, measured / elapsed, elapsed


def set_runtime_knobs(tf32, cufft_plan_cache):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    try:
        torch.backends.cuda.cufft_plan_cache.max_size = int(cufft_plan_cache)
        torch.backends.cuda.cufft_plan_cache.clear()
    except Exception:
        pass


def write_markdown(path, config, run_results):
    grouped = {}
    for r in run_results:
        key = (r.backend, r.tf32, r.cufft_plan_cache)
        grouped.setdefault(key, []).append(r.ms_per_decode)

    summary = []
    for key, vals in grouped.items():
        med = statistics.median(vals)
        mean = statistics.mean(vals)
        summary.append((med, mean, key, vals))
    summary.sort(key=lambda x: x[0])

    lines = []
    lines.append("# Phase 4 Torch Runtime Tuning Report")
    lines.append("")
    lines.append("## Config")
    for k, v in config.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Ranked Results (by median ms/decode)")
    for rank, (med, mean, key, vals) in enumerate(summary, start=1):
        backend, tf32, cache = key
        lines.append(
            f"{rank}. backend=`{backend}` tf32=`{tf32}` cufft_plan_cache=`{cache}` median_ms=`{med:.6f}` mean_ms=`{mean:.6f}` runs=`{[round(v, 6) for v in vals]}"  # noqa: E501
        )

    if summary:
        best = summary[0]
        b_backend, b_tf32, b_cache = best[2]
        lines.append("")
        lines.append("## Recommended Config")
        lines.append(
            f"- backend=`{b_backend}` tf32=`{b_tf32}` cufft_plan_cache=`{b_cache}` (lowest median ms/decode)"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_bool_list(s):
    out = []
    for t in s.split(','):
        v = t.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif v in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {t}")
    return out


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Phase 4 torch runtime tuning benchmark.")
    parser.add_argument("--nb-parts", type=int, default=100)
    parser.add_argument("--nb-machines", type=int, default=4)
    parser.add_argument("--inst-number", type=int, default=0)
    parser.add_argument("--decodes", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backends", default="torch_gpu,torch_gpu_unbatched")
    parser.add_argument("--tf32-values", default="true,false")
    parser.add_argument("--cufft-cache-values", default="32,128,256")
    parser.add_argument("--json-out", default="reports/phase4_torch_tuning.json")
    parser.add_argument("--md-out", default="reports/PHASE4_TORCH_TUNING.md")
    args = parser.parse_args()

    if args.warmup >= args.decodes:
        raise ValueError("--warmup must be smaller than --decodes")

    backend_names = [b.strip() for b in args.backends.split(',') if b.strip()]
    tf32_values = parse_bool_list(args.tf32_values)
    cache_values = parse_int_list(args.cufft_cache_values)

    instance_parts, instance_parts_unique = load_instance(args.nb_parts, args.nb_machines, args.inst_number)
    job_spec_all = pd.read_excel("data/PartsMachines/part-machine-information.xlsx", sheet_name="part", header=0, index_col=0)
    job_spec = job_spec_all.loc[instance_parts_unique]
    mach_spec = pd.read_excel("data/PartsMachines/part-machine-information.xlsx", sheet_name="machine", header=0, index_col=0)
    areas = pd.read_excel("data/PartsMachines/polygon_areas.xlsx", header=0)["Area"].tolist()
    thresholds = [t / args.nb_machines for t in range(1, args.nb_machines)]

    run_results = []
    for backend_name in backend_names:
        backend = create_collision_backend(backend_name)
        data = build_parts_data(args.nb_machines, instance_parts_unique, job_spec, mach_spec, areas, backend)

        for tf32 in tf32_values:
            for cache in cache_values:
                set_runtime_knobs(tf32, cache)
                for rep in range(args.repeats):
                    rng = np.random.default_rng(args.seed + rep)
                    chromosomes = rng.uniform(0.0, 1.0, size=(args.decodes, 2 * args.nb_parts)).astype(np.float32)
                    ms, dps, elapsed = evaluate_decodes(
                        data,
                        args.nb_parts,
                        args.nb_machines,
                        thresholds,
                        instance_parts,
                        backend,
                        chromosomes,
                        args.warmup,
                    )
                    run_results.append(
                        RunResult(
                            backend=backend.name,
                            tf32=tf32,
                            cufft_plan_cache=cache,
                            repeat=rep,
                            ms_per_decode=ms,
                            decodes_per_sec=dps,
                            total_seconds=elapsed,
                        )
                    )

    grouped = {}
    for r in run_results:
        key = (r.backend, r.tf32, r.cufft_plan_cache)
        grouped.setdefault(key, []).append(r.ms_per_decode)

    ranking = []
    for key, vals in grouped.items():
        ranking.append(
            {
                "backend": key[0],
                "tf32": key[1],
                "cufft_plan_cache": key[2],
                "median_ms_per_decode": statistics.median(vals),
                "mean_ms_per_decode": statistics.mean(vals),
                "all_ms_per_decode": vals,
            }
        )
    ranking.sort(key=lambda x: x["median_ms_per_decode"])

    payload = {
        "config": vars(args),
        "results": [asdict(r) for r in run_results],
        "ranking": ranking,
        "recommended": ranking[0] if ranking else None,
    }

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(md_path, vars(args), run_results)

    print(json.dumps(payload["recommended"], indent=2))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
