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
Example: Task Pinning in Torcalia

Description: This example demonstrates task pinning in Torcalia. Tasks can be pinned to specific MPI nodes, 
             ensuring they execute on the designated node. This is useful for scenarios where data locality or 
             resource affinity is important. 
             
Notes:
- The `node` argument controls which node a task is assigned to.
- When work stealing is enabled, pinned tasks can still be stolen by idle nodes unless `forced_node = true` is set, which guarantees the task 
  executes exclusively on the assigned node.

Three cases (all tasks are submitted to node 0):

Case 1 - Pinned to node 0, stealing disabled:
  Tasks remain on node 0, other nodes remain idle.

Case 2 - Pinned to node 0, stealing enabled:
  Other nodes can steal pending tasks from node 0.

Case 3 - Pinned to node 0, stealing enabled, forced_node = true:
  Tasks are guaranteed to execute on node 0, even though work stealing is enabled. 

Requirements: 2+ MPI processes, 1 worker thread per process.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2 task_pinning.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2,0 task_pinning.jl
"""

import Torcalia as Torc

nodes = Torc.num_nodes()

if nodes < 2
    println("[MPI Rank: $(Torc.node_id())] this example needs at least two MPI processes. exiting...")
    exit()
end

local_workers = Torc.num_local_workers()

if local_workers > 1
    println("[MPI Rank: $(Torc.node_id())] this example should use one Julia worker thread per MPI process. exiting...")
    exit()
end

function work(x::Int)::Int
    Torc.node_id() % 2 == 0 ? sleep(0.2) : nothing
    
    y = x * x
    
    println("Node $(Torc.node_id()) (worker thread $(Torc.worker_local_id())): work (input = $x, output = $y).")

    y
end

function main()
    ntasks = 10
    
    println("Scenario 1: Pinned to node 0, work stealing disabled (default).")

    tasks = [Torc.submit(work, i; node = 0) for i in 1:ntasks]
    Torc.wait(tasks)

    Torc.enable_stealing()
    println("\nScenario 2: Pinned to node 0, work stealing enabled and forced_node = false (tasks can be stolen).")

    tasks = [Torc.submit(work, i; node = 0) for i in 1:ntasks]
    Torc.wait(tasks)

    println("\nScenario 3: Pinned to node 0, work stealing enabled and forced_node = true (tasks cannot be stolen).")

    tasks = [Torc.submit(work, i; node = 0, forced_node = true) for i in 1:ntasks]
    Torc.wait(tasks)
end

Torc.init(main)