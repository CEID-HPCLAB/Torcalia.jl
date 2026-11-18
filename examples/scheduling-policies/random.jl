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
Example: Random Scheduling Policy in Torcalia

Description: This example demonstrates Torcalia's random scheduling policy, where tasks are assigned to MPI nodes in a non-deterministic manner.

Note: The random scheduling policy is set via `set_scheduling_policy(:RANDOM)`.

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3 random.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3,0 random.jl
"""

using Torcalia

function work(x::Int)::Int
    y = x^2
    
    println("Work: Input = $x, Output = $y \
             on worker thread $(Torcalia.worker_local_id()) of node $(Torcalia.node_id()).")

    y
end

function main()
    ntasks = 14
    
    inputs = 1:ntasks

    Torcalia.set_scheduling_policy(:RANDOM)

    tasks = [Torcalia.submit(work, x) for x in inputs]
    Torcalia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        
        y = Torcalia.result(task)
        
        println("Task $i: $(x)² = $y.")
    end
end

Torcalia.init(main)