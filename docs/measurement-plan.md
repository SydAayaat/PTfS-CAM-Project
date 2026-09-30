# PTfS CAM 2026: implementation and measurement notes

## What the slides require

The discretized equation is `A u = f`, with a matrix-free five-point stencil
for the negative Laplacian. The provided CG and symmetric Gauss-Seidel PCG
solvers seek the steady-state temperature field. Part 1 asks for OpenMP
parallelization, CG/PCG optimization, and correctness. Part 2 asks for a
time-based roofline model, measured kernel code balances, and scaling on
Fritz. There is no graded report submission in 2026; the exam covers the
project. A private experiment log is still useful for explaining every result.

## Model before collecting results

The reported `perf` metric is **one grid point per solver iteration** (LUP/s,
also IT/s). Do not sum the kernel throughputs: kernels execute in sequence.
For each kernel `k`, estimate its time per point as

`t_k/N >= max(B_k / 82e9, F_k / P_k)` seconds, then sum `t_k/N` over the
iteration. `B_k` is *bytes per point at the memory level being modeled*;
`F_k` is floating-point operations per point; `P_k` is the appropriate
compute ceiling. For Gauss-Seidel also account for the dependency/wavefront
limit and synchronization. The slides specify 82 GB/s for one 18-core
ccNUMA domain, not for the whole node.

| Per iteration | Kernel calls in current solver | Working model to verify |
| --- | --- | --- |
| CG | `APPLY_STENCIL`, `DOT_PRODUCT` (v·p), `UPDATE_RESIDUAL`, `AXPBY` (p update) | Model four sequential durations. |
| PCG | Same four, plus `GS_PRE_CON` and `DOT_PRODUCT` (r·z) | Model six sequential durations; SGS may be limited by dependencies rather than memory bandwidth. |

As a starting hypothesis, an in-place `AXPBY` on `p` has two double reads and
one double write (24 B/point), a streaming dot has two double reads (16),
and `UPDATE_RESIDUAL` has four double reads plus two double writes (48).
These are **traffic assumptions**, not measured DRAM balances: cache reuse,
write allocation, and Fritz's write-allocate avoidance change them. The
stencil reuses neighbor values, so count cache lines rather than five
independent DRAM loads. The forward/backward SGS sweeps need their own
measured or justified traffic model. Report a range when assumptions vary.

## Fritz run sheet

For the optional Moodle score, submit `leaderboard.sbatch` with `sbatch` from
the repository directory after loading the Intel compiler environment. It
requests one 72-core `singlenode` allocation, records output in
`leaderboard-<job-id>.txt`, runs the supplied tests, and invokes the unchanged
leaderboard script with one 72-CPU step. Add `--account=<authorized-account>`
to the `sbatch` command only if your course allocation requires it. The
script's output file is still submitted to Moodle manually.

1. Request a **dedicated Fritz node**; record job ID, node, compiler version,
   CPU frequency/governor, OpenMP settings, and the exact commit SHA. The
   course asks for 2.4 GHz and the performance governor.
2. Build with `CXX=icpx make` and run `OMP_NUM_THREADS=1 ./test`, then
   `OMP_NUM_THREADS=18 ./test`. Build and run `parallel_check` as well. A
   local GCC run is a correctness check, not a Fritz performance result.
3. Find the actual core IDs in each ccNUMA domain with `likwid-topology` and
   the site's CPU binding documentation. Bind 18 threads to one domain;
   use `OMP_PLACES=cores`, `OMP_PROC_BIND=close`, and a verified binding
   option from the local Slurm setup. Check placement before interpreting
   bandwidth. First touch in the Grid constructors follows the static row
   partition for large arrays.
4. Build **without LIKWID** for wall time. Run `./perf 2000 20000`,
   `./perf 20000 2000`, and `./perf 1000 400000`. Repeat runs and keep
   CG/PCG iterations, elapsed times, MLUP/s, and per-kernel timing summary.
   The last argument is the contiguous x direction. The largest case has
   400 million points: each double Grid occupies about 3.2 GB, and the
   eight live grids in PCG need about 25.6 GB plus overhead. Check the
   node allocation before running.
5. Rebuild with `LIKWID=on CXX=icpx make` for counters. Check available
   groups with `likwid-perfctr -a`, then use a suitable memory-traffic group,
   `-C` with the **verified** core list, and `-m` to collect the named code
   markers. Divide traffic for `APPLY_STENCIL` and `GS_PRE_CON` by their
   marker invocation counts times `ny*nx` to get bytes/LUP. Account for
   counter scope (socket/domain), repeated solver calls in `perf`, and any
   initialization markers. Collect timing runs separately because LIKWID
   markers and counters introduce overhead.
6. For 18, 36, 54, and 72 threads, use 1, 2, 3, and 4 ccNUMA domains on
   the *same dedicated node* with verified binding and memory placement.
   Record the run-to-run spread and compute `speedup = T18 / Tp` and
   `efficiency = speedup / (p/18)` for CG and PCG separately. Look at the
   timing summary to identify bandwidth saturation, SGS wavefront barriers,
   load imbalance, and NUMA traffic before proposing another change.

| ny | nx | N | 18-core CG/PCG | stencil B/LUP | SGS B/LUP | 18/36/54/72-core times |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 2000 | 20000 | 40,000,000 | measure | measure | measure | measure |
| 20000 | 2000 | 40,000,000 | measure | measure | measure | measure |
| 1000 | 400000 | 400,000,000 | measure | measure | measure | measure |

Different aspect ratios with the same `N` can change the row length, page
behavior, available wavefront tiles, and barrier overhead. The 400-million
point case also has 10 times as much work as the other two, so compare
throughput and code balance rather than raw elapsed time alone. The `perf`
program runs 20 solver iterations and resets the initial vector for PCG.

## Observations already verified outside Fritz

With GCC on an AMD EPYC virtual machine, the supplied 13 tests pass at 1 and
4 threads. The additional serial-reference check covers four rectangular
shapes at 1, 4, and 8 threads, including a repeat SGS application. Its
maximum absolute difference is below `6e-20`. Both solvers retain the
baseline iteration counts in the supplied tests (CG 724, PCG 247/251).
These are correctness observations, not predictions of Fritz throughput.
