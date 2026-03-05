# Computational Efficiency Roadmap

This roadmap operationalizes the 7 high-level recommendations into implementation phases we can execute one by one.

## Phase 0: GPU Backend Decision Gate (CuPy-first)
### Objective
Decide and lock the GPU backend before implementation phases that depend on kernel/runtime APIs.

### Why
Most downstream work (kernel implementation, batching, memory residency, streams, profiling) is backend-specific.

### Constraint
No planned use of learned components. This shifts default preference toward CuPy unless Torch demonstrates clearly better measured throughput or easier operational stability for this workload.

### Tasks
1. Define a narrow benchmark suite focused on your real collision workload (not generic FFT microbenchmarks).
2. Implement equivalent prototype kernels in CuPy and Torch for the same inputs.
3. Compare throughput, transfer overhead, memory footprint, implementation complexity, and debugging friction.
4. Record decision with pass/fail thresholds and lock backend for subsequent phases.

### Deliverables
- Backend decision record (`CuPy` or `Torch`) with benchmark evidence.
- Chosen runtime assumptions documented for all later phases.

### Success Criteria
- Decision made with reproducible measurements.
- No ambiguity left before Phase 1 implementation starts.

### Risks
- Choosing from non-representative benchmarks can mislead later architecture decisions.

---

## Phase 1: Hybrid Architecture (CPU orchestration + GPU collision kernels)
### Objective
Keep BRKGA orchestration on CPU while accelerating only geometric collision checks on GPU.

### Why
Decoding logic is branch-heavy and stateful; collision kernels are the most parallelizable hotspot.

### Tasks
1. Define architecture boundary: CPU handles population/evolution/decoder control flow; GPU handles overlap kernels.
2. Refactor collision checking into a backend interface (CPU backend + GPU backend).
3. Keep output equivalence tests to ensure fitness values match within tolerance.

### Deliverables
- `CollisionBackend` abstraction with pluggable CPU/GPU implementations.
- Config flag for backend selection.
- Correctness report (same instance, same seed, matching fitness).

### Success Criteria
- No behavior regression in decoded makespan.
- End-to-end throughput improves vs baseline on representative instances.

### Risks
- GPU acceleration underperforms if offload granularity is too small.

---

## Phase 2: Batch GPU Workloads
### Objective
Increase GPU efficiency by batching many collision checks per launch.

### Why
Unbatched kernels suffer from launch overhead and poor GPU occupancy.

### Tasks
1. Identify batchable collision operations across bins/parts/chromosomes.
2. Implement micro-batching queue with configurable batch size.
3. Add auto-tuning script to find throughput-optimal batch size.

### Deliverables
- Batched collision execution path.
- Batch-size benchmarking table.

### Success Criteria
- GPU path outperforms unbatched path significantly on medium/large workloads.

### Risks
- Large batches may increase memory pressure and latency spikes.

---

## Phase 3: Persistent Device Residency
### Objective
Minimize host-device transfer overhead by keeping reusable tensors on GPU.

### Why
Transfer overhead can erase GPU gains.

### Tasks
1. Move static assets (rotations, FFT kernels, metadata tensors) to device once.
2. Keep mutable bin state device-resident where feasible.
3. Replace sync-heavy calls with async pipelines and explicit synchronization points.

### Deliverables
- Device-memory lifecycle plan.
- Transfer profiling before/after.

### Success Criteria
- Host-device transfer time is a small fraction of total runtime.
- No repeated transfer of static data during generation loops.

### Risks
- Device memory fragmentation or OOM on larger instances.

---

## Phase 4: Backend Validation and Runtime Tuning
### Objective
Deep-tune the chosen backend runtime for this workload.

### Why
After backend lock-in, runtime details (streaming, memory pool behavior, FFT plans) determine real gains.

### Tasks
1. Expand benchmark suite to include realistic generation loops and mixed instance scales.
2. Tune backend-specific runtime settings (streams, memory pools, FFT plan reuse, sync points).
3. Validate performance stability across multiple runs and seeds.

### Deliverables
- Runtime tuning report for the chosen backend.
- Stable configuration defaults for later phases.

### Success Criteria
- Chosen backend reaches consistent high throughput without correctness drift.

### Risks
- Overfitting tuning to a single instance profile.

---

## Phase 5: Two-Tier Parallelism (CPU workers + GPU acceleration)
### Objective
Combine coarse CPU parallelism with fine GPU acceleration for better throughput.

### Why
Chromosome evaluations are parallel across individuals, while each decode has serial components.

### Tasks
1. Move from fixed thread pool to configurable process-based worker model for evaluation chunks.
2. Assign GPU work via stream-aware scheduler.
3. Prevent oversubscription (too many workers contending for one GPU).

### Deliverables
- Evaluation runtime with tunable `num_workers`, `chunk_size`, `gpu_streams`.
- Scaling curve (workers vs throughput).

### Success Criteria
- Throughput scales up to a stable optimum before plateau.
- No instability from contention or excessive context switching.

### Risks
- Multiprocessing + GPU context management complexity.

---

## Phase 6: Adaptive Collision Strategy (FFT only when beneficial)
### Objective
Use different collision kernels depending on part/bin size characteristics.

### Why
FFT is not always optimal, especially for small masks.

### Tasks
1. Add heuristic threshold policy (e.g., small -> direct bitset overlap, large -> FFT).
2. Implement both kernels behind one interface.
3. Tune threshold using benchmark grid search.

### Deliverables
- Adaptive kernel selector.
- Threshold tuning results.

### Success Criteria
- Adaptive mode beats FFT-only and direct-only baselines on mixed workloads.

### Risks
- Selector overhead or poor threshold generalization.

---

## Phase 7: Throughput-Optimized Evaluation Pipeline
### Objective
Optimize for evaluations-per-second, not single-evaluation latency.

### Why
BRKGA quality/time depends on total evaluations completed.

### Tasks
1. Build streaming pipeline: chromosome generation -> decode queue -> collision batches -> fitness return.
2. Add bounded queues for backpressure and stable memory usage.
3. Instrument pipeline stages (queue depth, stage times, GPU utilization).

### Deliverables
- Non-blocking evaluator pipeline.
- Live telemetry logging for throughput diagnostics.

### Success Criteria
- Higher sustained evaluations/sec across full generation loop.
- Stable runtime without queue blow-up.

### Risks
- Pipeline complexity may hinder debugging if observability is weak.

---

## Cross-Phase Guardrails
1. Keep deterministic mode (`seed`) for A/B performance comparisons.
2. Maintain a correctness harness to validate makespan equivalence per phase.
3. Benchmark with at least 3 instance scales (small/medium/large).
4. Record wall time, evals/sec, GPU utilization, and peak memory.
5. Only merge a phase when both correctness and throughput targets are met.

## Suggested Execution Order
1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7

## Definition of Done (Program Level)
1. End-to-end evaluator throughput improved materially on target workloads.
2. Fitness results remain consistent with baseline behavior.
3. Runtime architecture supports reproducible benchmarking and future tuning.
