# Phase 0 Backend Decision Report

## Benchmark Configuration
- Parts directory: `data/partsMatrices`
- Sample parts: `24`
- Part selection: `largest`
- Bin size: `768x768`
- Densities: `[0.05, 0.2, 0.4]`
- Iterations: `12`
- Warmup: `3`
- Total generated cases: `72`

## Results
- `torch`: available=`True`, device=`NVIDIA GeForce GTX 1650 Ti`, ms_per_case_iter=`9.872388`, throughput_case_iters_per_sec=`101.29`, notes=`ok`
- `cupy`: available=`True`, device=`NVIDIA GeForce GTX 1650 Ti`, ms_per_case_iter=`11.204338`, throughput_case_iters_per_sec=`89.25`, notes=`ok`

## Decision
- Chosen backend: `torch`
- Rationale: Torch significantly faster than CuPy on measured workload.

## Next Step
- Implement Phase 1 against the chosen backend and keep CPU backend for parity checks.