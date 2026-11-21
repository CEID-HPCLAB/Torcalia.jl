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
Example: Single Program Multiple Data (SPMD) Execution Model in Torcalia

Description: This example demonstrates Torcalia's SPMD execution model (spmd function), where the same task function runs on all MPI nodes. 

Workflow: Each MPI node executes the same function `work`, but with different data based on its MPI rank (node ID).

Requirements: 2+ MPI processes, 1+ worker thread(s) per process

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2 spmd.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2,0 spmd.jl
"""

import Torcalia as Torc

nodes = Torc.num_nodes()

if nodes < 2
    println("[MPI Rank: $(Torc.node_id())] this example needs at least two MPI processes. exiting...")
    exit()
end

function work(; multiplier::Int)::Tuple{Int, Int}
    node_id = Torc.node_id()
    
    A = [(node_id * 10 + i) * multiplier for i in 1:node_id]
    
    sum(A), node_id
end

function main()
    println("Running with $(Torc.num_nodes()) MPI nodes.")
    
    tasks = Torc.spmd(work; multiplier = 2)
    
    for (i, task) in enumerate(tasks)
        res = Torc.result(task)
        
        println("Node $(res[2]): sum(A) = $(res[1]).")
    end

    println("All nodes completed their assigned task successfully.")
end

Torc.init(main)