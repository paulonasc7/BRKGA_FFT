# Phase 2 Batching Benchmark Report

## Config
- nb_parts: `50`
- nb_machines: `2`
- inst_number: `0`
- decodes: `18`
- warmup: `3`
- seed: `42`
- json_out: `reports\phase2_batching_benchmark_P50M2-0.json`
- md_out: `reports\PHASE2_BATCHING_REPORT_P50M2-0.md`

## Results
- `torch_cuda_unbatched`: ms_per_decode=`267.630800`, decodes_per_sec=`3.736`, total_seconds=`4.014`
- `torch_cuda_batched`: ms_per_decode=`304.755387`, decodes_per_sec=`3.281`, total_seconds=`4.571`

## Speedup
- Batched speedup vs unbatched (ms/decode): `0.878x`