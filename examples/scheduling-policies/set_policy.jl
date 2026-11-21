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
Example: Configuring and Using Different Scheduling Policies in Torcalia

Description: This example demonstrates how to configure and use different scheduling policies in Torcalia, including the default round-robin policy, 
             weighted round-robin (WRR), local scheduling, random scheduling, and a custom scheduling policy based on task IDs. The custom policy 
             assigns tasks with even IDs to MPI nodes with even ranks and tasks with odd IDs to MPI nodes with odd ranks, randomly selecting a node 
             from the corresponding group. 

Notes: Regarding the WRR scheduling policy, the weights are specified as a vector of integers, where each integer represents the weight 
       of the corresponding node. For example, if there are 4 MPI nodes and the weights are set to [1, 1, 2, 3], then node 0 and node 1 
       will receive tasks in a 1:1 ratio, node 2 will receive tasks in a 2:1 ratio compared to nodes 0 and 1, and node 3 will 
       receive tasks in a 3:1 ratio compared to nodes 0 and 1. 

Requirements: 2+ MPI processes, 1+ worker threads per MPI process.

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3 set_policy.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3,0 set_policy.jl
"""

import Torcalia as Torc

nodes = Torc.num_nodes()

if nodes < 2
    println("[MPI Rank: $(Torc.node_id())] this example needs at least two MPI processes. exiting...")
    exit()
end

function work(x::Int)::Int
    node_id = Torc.node_id()

    local_worker_id = Torc.worker_local_id()

    sleep(0.5)
    
    y = x^2

    println("[MPI Rank: $node_id, Worker thread local ID: $local_worker_id] Work: input = $x, output = $y.")

    y
end

function custom_scheduling_policy(task_id::Int)::Int
    candidates = iseven(task_id) ? (0:2:nodes-1) : (1:2:nodes-1)

    rand(candidates)
end

function main()
    ntasks = 20
    
    inputs = 1:ntasks

    println("[MPI Rank: $(Torc.node_id())] Submitting $ntasks tasks. Scheduling policy: $(Torc.get_scheduling_policy()).")

    tasks = [Torc.submit(work, x) for x in inputs]
    
    Torc.wait(tasks)

    @assert all(Torc.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    weights = [1, 1, 2, 3]

    Torc.set_scheduling_policy(:WRR, weights)

    println("\n[MPI Rank: $(Torc.node_id())] Submitting $ntasks tasks. Scheduling policy: \
             $(Torc.get_scheduling_policy()) with weights $weights.")

    tasks = [Torc.submit(work, x) for x in inputs]
    
    Torc.wait(tasks)

    @assert all(Torc.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    Torc.set_scheduling_policy(:LOCAL)

    println("\n[MPI Rank: $(Torc.node_id())] Submitting $ntasks tasks. Scheduling policy: $(Torc.get_scheduling_policy()).")

    tasks = [Torc.submit(work, x) for x in inputs]
    
    Torc.wait(tasks)

    @assert all(Torc.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    Torc.set_scheduling_policy(:RANDOM)

    println("\n[MPI Rank: $(Torc.node_id())] Submitting $ntasks tasks. Scheduling policy: $(Torc.get_scheduling_policy()).")

    tasks = [Torc.submit(work, x) for x in inputs]
    
    Torc.wait(tasks)

    @assert all(Torc.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    println("\n[MPI Rank: $(Torc.node_id())] Submitting $ntasks tasks with custom scheduling policy.")

    tasks = [Torc.submit(work, x; distribution_policy = (custom_scheduling_policy, x)) for x in inputs]

    Torc.wait(tasks)

    @assert all(Torc.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))
end

Torc.init(main)