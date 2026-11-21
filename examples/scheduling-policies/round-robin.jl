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
Example: Round-Robin Scheduling Policy in Torcalia

Description: This example demonstrates Torcalia's default round-robin scheduling policy, where tasks 
             are distributed evenly across available MPI nodes in a cyclic manner.

Notes:
- The scheduling policy is set via `set_scheduling_policy(:ROUND_ROBIN)`.
- Round-Robin is the default task distribution policy in Torcalia.

In this example, 12 tasks are submitted to compute the square of integers from 1 to 12. The tasks are distributed 
across 4 MPI nodes in a round-robin fashion, resulting in the following distribution:

    Cycle 1 (tasks 1-4)     Cycle 2 (tasks 5-8)     Cycle 3 (tasks 9-12)
    ────────────────────    ────────────────────    ────────────────────
    node 0 → [1]            node 0 → [5]            node 0 → [ 9]
    node 1 → [2]            node 1 → [6]            node 1 → [10]
    node 2 → [3]            node 2 → [7]            node 2 → [11]
    node 3 → [4]            node 3 → [8]            node 3 → [12]

Final distribution of tasks across MPI nodes:

    ┌────────┬──────────────┬─────────┬──────┐
    │  Node  │    Tasks     │  Count  │   %  │
    ├────────┼──────────────┼─────────┼──────┤
    │ node 0 │ [1, 5,  9]   │    3    │  25% │
    │ node 1 │ [2, 6, 10]   │    3    │  25% │
    │ node 2 │ [3, 7, 11]   │    3    │  25% │
    │ node 3 │ [4, 8, 12]   │    3    │  25% │
    └────────┴──────────────┴─────────┴──────┘

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2 round-robin.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2,0 round-robin.jl
"""

import Torcalia as Torc

function work(x::Int)::Int
    y = x^2
    
    println("Work: Input = $x, Output = $y \
             on worker thread $(Torc.worker_local_id()) of node $(Torc.node_id()).")

    y
end

function main()
    ntasks = 12
    
    inputs = 1:ntasks

    Torc.set_scheduling_policy(:ROUND_ROBIN)

    tasks = [Torc.submit(work, x) for x in inputs]
    Torc.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        
        y = Torc.result(task)
        
        println("Task $i: $(x)² = $y.")
    end
end

Torc.init(main)