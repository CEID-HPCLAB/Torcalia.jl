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
Example: Master-Worker Paradigm in Torcalia

Description: This example demonstrates the fundamental master-worker paradigm in Torcalia. The master process (rank 0) creates and submits tasks, 
             then waits for their completion. While waiting, the master's main thread actively participates in executing tasks rather than remaining idle.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2 master_worker.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2,0 master_worker.jl
"""

using Torcalia

function work(x::Int)::Int
    sleep(1)
    
    y = x^2

    println("Work: input = $x, output = $y \
             on worker thread $(Torcalia.worker_local_id()) of node $(Torcalia.node_id()).")
    y
end

function main()
    ntasks = 8
    
    inputs = 1:ntasks

    println("Submitting $ntasks tasks.")

    t0 = Torcalia.gettime()

    tasks = [Torcalia.submit(work, x) for x in inputs]
    Torcalia.wait(tasks)

    elapsed = Torcalia.gettime() - t0
    
    for (i, task) in enumerate(tasks)
        y = Torcalia.result(task)
        println("Task $i: $(inputs[i])² = $y.")
    end

    println("All tasks completed successfully.")
    println("Total elapsed time: $(round(elapsed, digits = 3)) seconds.")
end

Torcalia.init(main)