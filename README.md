# Programming Techniques for Supercomputers - CAM project

## Background: Modelling 2D steady-state heat equation
This code project for the 10 ECTS PTfS module solves a linear system of equations to find a steady-state temperature distribution on a rectangular plate using the Conjugate Gradient (CG) and Preconditioned Conjugate Gradient (PCG) with symmetric Gauss-Seidel preconditioning methods.

Please see the tutorials for more information.

## Project work and verification

The 2026 project has no graded report submission. The 10 ECTS exam includes
questions on the implementation and measurements; the Moodle leaderboard is
optional (the slides give September 30 as its closing date). See
[`docs/measurement-plan.md`](docs/measurement-plan.md) for the experiment
matrix, roofline method, and the distinction between local checks and Fritz
results.

The optimized implementation parallelizes the independent grid operations and
uses a tiled diagonal wavefront for symmetric Gauss-Seidel. It keeps the
forward/backward point order inside each tile, and parallel tiles have no
dependencies on one another. CG/PCG update the solution and residual and
compute the residual norm in one traversal. The 32768-point threshold avoids
starting a thread team for small grids. For a correctness check that exercises
rectangular grids and the parallel wavefront, run:

```bash
CXX=g++ make
OMP_NUM_THREADS=4 ./test
CXX=g++ make parallel_check
./parallel_check
```

The `parallel_check` program sets 1, 4, and 8 threads itself. Compile with
`CXX=icpx` on Fritz and rerun both checks before benchmarking. The ordinary
`make` target cleans generated executables and object files first.

## Optional Moodle leaderboard run on Fritz

The batch file requests one 72-core `singlenode` node, builds with `icpx`,
runs `./test`, then invokes the original leaderboard script with OpenMP
settings and a 72-CPU Slurm step. Load the Intel compiler module required by
your course account before submitting. If your account requires an explicit
Slurm account, pass it with `sbatch --account=<authorized-account>`.

```bash
sbatch leaderboard.sbatch
squeue -u "$USER"
cat leaderboard-<job-id>.txt
```

The output file contains the exact leaderboard command and the score/token
printed by `verify`; submit that result and command manually in Moodle. The
leaderboard script itself remains unchanged because its integrity check hashes
that file.

## Usage
1) Compile program by calling
    ```bash
    CXX=<compiler icpx or g++> make
    ```

    To print additional debug output during the execution, compile with `-DDEBUG`:
   ```bash
   CXX=<compiler icpx or g++> make EXTRA_FLAGS=-DDEBUG
   ```

   To compile with LIKWID use:
   ```bash
   LIKWID=on CXX=<compiler icpx or g++> make
   ```

1)  Run tests by running
    ```bash
    ./test
    ```

2)  Run performance tests by running
    ```bash
    ./perf <grid size y> <grid size x>
    ```

3) Make plots (only when compiled with `--DDEBUG`):
    ```bash
    ./test
    ./plot.sh
    ```
    This generates some `.png` files showing you calculations. As a reference, we provide images in the folder resources.
