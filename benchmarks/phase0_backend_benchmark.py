import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import cupy as cp
except Exception:
    cp = None


@dataclass
class BackendResult:
    backend: str
    available: bool
    device: str
    cases: int
    iterations: int
    warmup: int
    total_seconds: float
    ms_per_case_iter: float
    gops_like: float
    notes: str


def load_part_matrices(parts_dir: Path, sample_parts: int, part_selection: str):
    files = sorted(parts_dir.glob('matrix_*.npy'))
    if not files:
        raise FileNotFoundError(f'No part matrices found in {parts_dir}')

    if part_selection == 'largest':
        scored = []
        for p in files:
            arr = np.load(p, mmap_mode='r')
            area = int(arr.shape[0] * arr.shape[1])
            scored.append((area, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [p for _, p in scored[:sample_parts]]
    else:
        selected = files[:sample_parts]

    matrices = [np.load(p).astype(np.float32) for p in selected]
    return matrices


def build_cases(part_mats, bin_length, bin_width, densities):
    rng = np.random.default_rng(1234)
    cases = []
    for part in part_mats:
        ph, pw = part.shape
        if ph > bin_length or pw > bin_width:
            continue

        part_flipped = np.flip(part, axis=(0, 1))
        part_padded = np.pad(part_flipped, ((0, bin_length - ph), (0, bin_width - pw)), mode='constant')

        for d in densities:
            grid = (rng.random((bin_length, bin_width), dtype=np.float32) < d).astype(np.float32)
            cases.append((grid, part_padded, ph, pw))
    return cases


def torch_benchmark(cases, iterations, warmup):
    if torch is None:
        return BackendResult('torch', False, 'n/a', 0, iterations, warmup, 0.0, 0.0, 0.0, 'torch not installed')
    if not torch.cuda.is_available():
        return BackendResult('torch', False, 'n/a', 0, iterations, warmup, 0.0, 0.0, 0.0, 'torch CUDA unavailable')

    device = torch.device('cuda:0')
    device_name = torch.cuda.get_device_name(0)

    gpu_cases = []
    for grid, part_padded, ph, pw in cases:
        g = torch.tensor(grid, dtype=torch.float32, device=device)
        p = torch.tensor(part_padded, dtype=torch.float32, device=device)
        pfft = torch.fft.fft2(p)
        gpu_cases.append((g, pfft, ph, pw))

    def run_once():
        feasible = 0
        for g, pfft, ph, pw in gpu_cases:
            gfft = torch.fft.fft2(g)
            conv = torch.fft.ifft2(gfft * pfft).real[ph - 1 : g.shape[0], pw - 1 : g.shape[1]]
            if (torch.round(conv) == 0).any():
                feasible += 1
        return feasible

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_ops = len(gpu_cases) * iterations
    ms_per = (elapsed / total_ops) * 1000 if total_ops else 0.0
    gops_like = total_ops / elapsed if elapsed > 0 else 0.0

    return BackendResult('torch', True, device_name, len(gpu_cases), iterations, warmup, elapsed, ms_per, gops_like, 'ok')


def cupy_benchmark(cases, iterations, warmup):
    if cp is None:
        return BackendResult('cupy', False, 'n/a', 0, iterations, warmup, 0.0, 0.0, 0.0, 'cupy not installed')

    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
    except Exception:
        return BackendResult('cupy', False, 'n/a', 0, iterations, warmup, 0.0, 0.0, 0.0, 'cupy device unavailable')

    gpu_cases = []
    for grid, part_padded, ph, pw in cases:
        g = cp.asarray(grid, dtype=cp.float32)
        p = cp.asarray(part_padded, dtype=cp.float32)
        pfft = cp.fft.fft2(p)
        gpu_cases.append((g, pfft, ph, pw))

    def run_once():
        feasible = 0
        for g, pfft, ph, pw in gpu_cases:
            gfft = cp.fft.fft2(g)
            conv = cp.fft.ifft2(gfft * pfft).real[ph - 1 : g.shape[0], pw - 1 : g.shape[1]]
            if cp.any(cp.rint(conv) == 0):
                feasible += 1
        return feasible

    for _ in range(warmup):
        run_once()
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        run_once()
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

    total_ops = len(gpu_cases) * iterations
    ms_per = (elapsed / total_ops) * 1000 if total_ops else 0.0
    gops_like = total_ops / elapsed if elapsed > 0 else 0.0

    return BackendResult('cupy', True, device_name, len(gpu_cases), iterations, warmup, elapsed, ms_per, gops_like, 'ok')


def decide(results):
    torch_r = next((r for r in results if r.backend == 'torch'), None)
    cupy_r = next((r for r in results if r.backend == 'cupy'), None)

    if cupy_r and cupy_r.available and torch_r and torch_r.available:
        # CuPy-preferred if performance is close or better (<=10% slower).
        if cupy_r.ms_per_case_iter <= torch_r.ms_per_case_iter * 1.10:
            return 'cupy', 'CuPy preferred by policy (no ML planned) and performance within threshold.'
        return 'torch', 'Torch significantly faster than CuPy on measured workload.'

    if cupy_r and cupy_r.available:
        return 'cupy', 'Only CuPy backend available.'
    if torch_r and torch_r.available:
        return 'torch', 'Only Torch backend available (provisional decision; install CuPy to complete comparison).'
    return 'none', 'No usable GPU backend available.'


def write_markdown(path: Path, args, cases_count, results, chosen_backend, rationale):
    lines = []
    lines.append('# Phase 0 Backend Decision Report')
    lines.append('')
    lines.append('## Benchmark Configuration')
    lines.append(f'- Parts directory: `{args.parts_dir}`')
    lines.append(f'- Sample parts: `{args.sample_parts}`')
    lines.append(f'- Part selection: `{args.part_selection}`')
    lines.append(f'- Bin size: `{args.bin_length}x{args.bin_width}`')
    lines.append(f'- Densities: `{args.densities}`')
    lines.append(f'- Iterations: `{args.iterations}`')
    lines.append(f'- Warmup: `{args.warmup}`')
    lines.append(f'- Total generated cases: `{cases_count}`')
    lines.append('')
    lines.append('## Results')
    for r in results:
        lines.append(f"- `{r.backend}`: available=`{r.available}`, device=`{r.device}`, ms_per_case_iter=`{r.ms_per_case_iter:.6f}`, throughput_case_iters_per_sec=`{r.gops_like:.2f}`, notes=`{r.notes}`")
    lines.append('')
    lines.append('## Decision')
    lines.append(f'- Chosen backend: `{chosen_backend}`')
    lines.append(f'- Rationale: {rationale}')
    lines.append('')
    lines.append('## Next Step')
    lines.append('- Implement Phase 1 against the chosen backend and keep CPU backend for parity checks.')

    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Phase 0 backend benchmark for collision workload.')
    parser.add_argument('--parts-dir', default='data/partsMatrices')
    parser.add_argument('--sample-parts', type=int, default=16)
    parser.add_argument('--part-selection', choices=['first', 'largest'], default='first')
    parser.add_argument('--bin-length', type=int, default=256)
    parser.add_argument('--bin-width', type=int, default=256)
    parser.add_argument('--densities', type=float, nargs='+', default=[0.05, 0.15, 0.30])
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--json-out', default='reports/phase0_backend_benchmark.json')
    parser.add_argument('--md-out', default='reports/PHASE0_BACKEND_DECISION.md')
    args = parser.parse_args()

    part_mats = load_part_matrices(Path(args.parts_dir), args.sample_parts, args.part_selection)
    cases = build_cases(part_mats, args.bin_length, args.bin_width, args.densities)

    if not cases:
        raise RuntimeError('No benchmark cases were generated. Increase bin size or reduce sample parts.')

    results = [
        torch_benchmark(cases, args.iterations, args.warmup),
        cupy_benchmark(cases, args.iterations, args.warmup),
    ]

    chosen_backend, rationale = decide(results)

    payload = {
        'config': vars(args),
        'cases': len(cases),
        'results': [asdict(r) for r in results],
        'decision': {
            'backend': chosen_backend,
            'rationale': rationale,
        },
    }

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(md_path, args, len(cases), results, chosen_backend, rationale)

    print(json.dumps(payload['decision'], indent=2))
    print(f'Wrote: {json_path}')
    print(f'Wrote: {md_path}')


if __name__ == '__main__':
    main()
