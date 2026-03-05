# Phase 0 Backend Decision Report

## Benchmark Configuration
- Parts directory: `data/partsMatrices`
- Sample parts: `16`
- Part selection: `largest`
- Bin size: `1024x1024`
- Densities: `[0.05, 0.2]`
- Iterations: `8`
- Warmup: `2`
- Total generated cases: `32`

## Results
- `torch`: available=`True`, device=`NVIDIA GeForce GTX 1650 Ti`, ms_per_case_iter=`16.738514`, throughput_case_iters_per_sec=`59.74`, notes=`ok`
- `cupy`: available=`True`, device=`NVIDIA GeForce GTX 1650 Ti`, ms_per_case_iter=`20.053145`, throughput_case_iters_per_sec=`49.87`, notes=`ok`

## Decision
- Chosen backend: `torch`
- Rationale: Torch significantly faster than CuPy on measured workload.

## Next Step
- Implement Phase 1 against the chosen backend and keep CPU backend for parity checks.