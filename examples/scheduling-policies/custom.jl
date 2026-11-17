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
Example: Custom Scheduling Policies in Torcalia

Description: This example demonstrates Torcalia's support for custom scheduling policies, where users can define their own task distribution strategies.

Note: 
- The custom scheduling policy is set via the `distribution_policy` argument in `submit()`.
- The custom scheduling function must return a valid node index (0 to num_nodes - 1). If an invalid value is returned, Torcalia falls back to Round-Robin.

In this example, 12 tasks are submitted to compute the square of integers from 1 to 12. The custom scheduling function distributes tasks across 4 MPI nodes
based on their index, assigning even-indexed tasks to nodes 0 and 1, and odd-indexed tasks to nodes 2 and 3 in a round-robin fashion, 
resulting in the following distribution:

    even tasks (2,4,6,...)    odd tasks (1,3,5,...)
    ──────────────────────    ─────────────────────
    node 0 → [2, 6, 10]       node 2 → [1, 5, 9]
    node 1 → [4, 8, 12]       node 3 → [3, 7, 11]

Final distribution of tasks across MPI nodes:

    ┌────────┬──────────────┬─────────┬──────┐
    │  Node  │    Tasks     │  Count  │   %  │
    ├────────┼──────────────┼─────────┼──────┤
    │ node 0 │ [2, 6, 10]   │    3    │  25% │
    │ node 1 │ [4, 8, 12]   │    3    │  25% │
    │ node 2 │ [1, 5, 9]    │    3    │  25% │
    │ node 3 │ [3, 7, 11]   │    3    │  25% │
    └────────┴──────────────┴─────────┴──────┘

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3 custom.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3,0 custom.jl
"""

using Torcalia

function work(x::Int)::Int
    y = x^2
    
    println("Work: Input = $x, Output = $y \
             on worker thread $(Torcalia.worker_local_id()) of node $(Torcalia.node_id()).")

    y
end

function custom_scheduler(task_id::Int)::Int
    if iseven(task_id)
        return mod(task_id ÷ 2 - 1, 2)
    end

    2 + mod((task_id - 1) ÷ 2, 2)
end

function main()
    ntasks = 12
    
    inputs = 1:ntasks
    
    tasks = [Torcalia.submit(work, x; distribution_policy = (custom_scheduler, x)) for x in inputs]
    
    Torcalia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        
        y = Torcalia.result(task)
        
        println("Task $i: $(x)² = $y.")
    end
end

Torcalia.init(main)