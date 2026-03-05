# Phase 0 Backend Decision Report

## Benchmark Configuration
- Parts directory: `data/partsMatrices`
- Sample parts: `16`
- Bin size: `256x256`
- Densities: `[0.05, 0.15, 0.3]`
- Iterations: `30`
- Warmup: `5`
- Total generated cases: `45`

## Results
- `torch`: available=`True`, device=`NVIDIA GeForce GTX 1650 Ti`, ms_per_case_iter=`0.628394`, throughput_case_iters_per_sec=`1591.36`, notes=`ok`
- `cupy`: available=`True`, device=`NVIDIA GeForce GTX 1650 Ti`, ms_per_case_iter=`0.785084`, throughput_case_iters_per_sec=`1273.75`, notes=`ok`

## Decision
- Chosen backend: `torch`
- Rationale: Torch significantly faster than CuPy on measured workload.

## Next Step
- Implement Phase 1 against the chosen backend and keep CPU backend for parity checks.