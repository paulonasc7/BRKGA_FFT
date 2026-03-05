# Phase 0 Backend Decision Report

## Benchmark Configuration
- Parts directory: `data/partsMatrices`
- Sample parts: `24`
- Part selection: `largest`
- Bin size: `512x512`
- Densities: `[0.05, 0.2, 0.4]`
- Iterations: `20`
- Warmup: `4`
- Total generated cases: `72`

## Results
- `torch`: available=`True`, device=`NVIDIA GeForce GTX 1650 Ti`, ms_per_case_iter=`4.419610`, throughput_case_iters_per_sec=`226.26`, notes=`ok`
- `cupy`: available=`True`, device=`NVIDIA GeForce GTX 1650 Ti`, ms_per_case_iter=`4.614485`, throughput_case_iters_per_sec=`216.71`, notes=`ok`

## Decision
- Chosen backend: `cupy`
- Rationale: CuPy preferred by policy (no ML planned) and performance within threshold.

## Next Step
- Implement Phase 1 against the chosen backend and keep CPU backend for parity checks.