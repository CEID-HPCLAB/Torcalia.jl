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
Example: Work Stealing in Torcalia

Description: This example demonstrates Torcalia's work stealing mechanism with imbalanced workloads 
             (workloads with irregular parallelism) across MPI nodes.

Notes:
- Even-ranked nodes (0, 2, 4, ...): Execute tasks with 0.2s delay (artificial bottleneck)
- Odd-ranked nodes (1, 3, 5, ...): Execute tasks immediately

Case 1 - Work Stealing Enabled:
    Tasks are distributed via round-robin. Idle workers can steal pending tasks from overloaded nodes, achieving dynamic (adaptive) load balancing.

Case 2 - Work Stealing Disabled:
    Tasks are distributed via round-robin but work stealing is disabled, so tasks remain on their initially assigned nodes without dynamic redistribution. 
    Therefore, the bottleneck from tasks assigned to even-ranked nodes becomes more pronounced, resulting in longer overall execution time.

Requirements: 2+ MPI processes, 1 worker thread per process.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2 work_stealing.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2,0 work_stealing.jl
"""

using Torcalia

function work(x::Int)::Int
    Torcalia.node_id() % 2 == 0 ? sleep(0.2) : nothing
    
    y = x^4
    
    println("Node $(Torcalia.node_id()) (worker thread $(Torcalia.worker_local_id())): work (input = $x, output = $y).")
    
    y
end

function main()
    nodes = Torcalia.num_nodes()

    if nodes < 2
        println("[MPI Rank: $(Torcalia.node_id())] this example needs at least two MPI processes. exiting...")
        exit()
    end

    local_workers = Torcalia.num_local_workers()

    if local_workers > 1
        println("[MPI Rank: $(Torcalia.node_id())] this example should use one Julia worker thread per MPI process. exiting...")
        exit()
    end
    
    ntasks = 40
    
    Torcalia.enable_stealing()

    println("Case 1: Submitting $ntasks tasks with work stealing enabled.")

    t0 = Torcalia.gettime()
    
    tasks = Torcalia.Future[]
    
    for i in 1:ntasks
        push!(tasks, Torcalia.submit(work, i))
    end

    Torcalia.wait(tasks)
    
    elapsed = Torcalia.gettime() - t0

    for (i, task) in enumerate(tasks)
        println("Task $i: $(Torcalia.input(task)...)^4 = $(Torcalia.result(task)).")
    end

    println("Total elapsed time (Case 1): $(round(elapsed, digits = 3)) seconds.")

    Torcalia.disable_stealing()

    println("\nCase 2: Submitting $ntasks tasks with work stealing disabled.")

    t0 = Torcalia.gettime()
    
    tasks = Torcalia.Future[]
    
    for i in 1:ntasks
        push!(tasks, Torcalia.submit(work, i))
    end

    Torcalia.wait(tasks)
    
    elapsed = Torcalia.gettime() - t0

    for (i, task) in enumerate(tasks)
        println("Task $i: $(Torcalia.input(task)...)^4 = $(Torcalia.result(task)).")
    end

    println("Total elapsed time (Case 2): $(round(elapsed, digits = 3)) seconds.")
end

Torcalia.init(main)