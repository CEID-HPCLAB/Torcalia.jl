#=
 * ==================================================================================
 * Torcalia.jl — An MPI-based Runtime System for Asynchronous Task Parallelism in Julia
 * ==================================================================================
 * This file is part of Torcalia, an MPI-based runtime system for asynchronous
 * many task (AMT) parallelism in Julia across both shared and distributed memory
 * platforms, developed at the High Performance Computing Laboratory (HPCLAB)
 * at the Computer Engineering & Informatics Department (CEID),
 * University of Patras, Greece.
 *
 * Repository: https://github.com/CEID-HPCLAB/Torcalia.jl
 *
 * Authors: Argiris Sofotasios (CEID-HPCLAB)
 * License: MIT License — see the LICENSE file in the repository root.
 * ==================================================================================
=#

"""
Example: 2D Shared-Memory MPI Windows in Torcalia

Description: This example demonstrates how 2D MPI shared-memory windows can be used on top of Torcalia. 
             Detailed description regarding the integration of MPI shared-memory windows into Torcalia
             can be found in `shm_win.jl`.

Requirements: 2+ MPI processes, 1+ worker thread(s) per process.

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2 shm_win_2d.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2,0 shm_win_2d.jl
"""

using Torcalia

function work(; A::Matrix{Int}, row::Int)
    old_vals = A[row, :]
    
    for j in 1:size(A, 2)
        A[row, j] += 5 * row + j
    end
    
    new_vals = A[row, :]
    
    println("[MPI Rank: $(Torcalia.node_id())] Worker thread $(Torcalia.worker_local_id()) of node $(Torcalia.node_id()) \
             updated row $row from $old_vals to $new_vals.")
end

function main()
    nodes = Torcalia.num_nodes()

    if nodes < 2
        println("[MPI Rank: $(Torcalia.node_id())] this example needs at least two MPI processes. exiting...")
        exit()
    end

    nrows = Int(nodes); ncols = 4
    
    A = Torcalia.shw_alloc((nrows, ncols), Int)

    println("[MPI Rank: $(Torcalia.node_id())] Allocated 2D MPI shared-memory array A ($nrows x $ncols, Int).")

    if Torcalia.node_id() == 0
        for i in 1:nrows, j in 1:ncols
            A[i, j] = 100 * (i - 1) + j
        end
    end

    println("[MPI Rank: $(Torcalia.node_id())] Initial values of A: $A.")

    tasks = [Torcalia.submit(work; A = A, row = i, node = i-1) for i in 1:nrows]

    Torcalia.wait(tasks)

    println("[MPI Rank: $(Torcalia.node_id())] Final values of A after all tasks completed: $A.")

    @assert A == [100 * (i - 1) + 5 * i + 2 * j for i in 1:nrows, j in 1:ncols]
end

Torcalia.init(main)