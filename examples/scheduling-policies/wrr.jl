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
Example: Weighted Round-Robin (WRR) Scheduling Policy in Torcalia

Description: This examples demonstrates Torcalia's weighted round-robin scheduling policy, where tasks are distributed across nodes 
             proportionally based on user-defined weights. This approach is particularly useful in heterogeneous computing environments.

Notes:
- The scheduling policy is set via `set_scheduling_policy(:WRR, weights)`, which propagates the policy and weights to all MPI nodes.
- The weights are specified as a vector of integers, where each integer represents the weight of the corresponding node.

In this example, 14 tasks are submitted to compute the square of integers from 1 to 14. The tasks are distributed
across 4 MPI nodes with weights [1, 1, 2, 3], resulting in the following distribution:

    Cycle 1 (tasks  1- 7)        Cycle 2 (tasks  8-14)
    ─────────────────────        ─────────────────────
    node 0 (w=1) → [1]           node 0 (w=1) → [8]
    node 1 (w=1) → [2]           node 1 (w=1) → [9]
    node 2 (w=2) → [3, 4]        node 2 (w=2) → [10, 11]
    node 3 (w=3) → [5, 6, 7]     node 3 (w=3) → [12, 13, 14]

Final distribution of tasks across MPI nodes:

    ┌────────┬────────┬────────────────────────┬─────────┬──────┐
    │  Node  │ Weight │          Tasks         │  Count  │   %  │
    ├────────┼────────┼────────────────────────┼─────────┼──────┤
    │ node 0 │   1    │ [1, 8]                 │    2    │  14% │
    │ node 1 │   1    │ [2, 9]                 │    2    │  14% │
    │ node 2 │   2    │ [3, 4, 10, 11]         │    4    │  29% │
    │ node 3 │   3    │ [5, 6, 7, 12, 13, 14]  │    6    │  43% │
    └────────┴────────┴────────────────────────┴─────────┴──────┘

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3 weighted-round-robin.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3,0 weighted-round-robin.jl
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
    
    weights = [1, 1, 2, 3]

    Torcalia.set_scheduling_policy(:WRR,  weights)

    tasks = [Torcalia.submit(work, x) for x in inputs]
    Torcalia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        
        y = Torcalia.result(task)
        
        println("Task $i: $(x)² = $y.")
    end
end

Torcalia.init(main)