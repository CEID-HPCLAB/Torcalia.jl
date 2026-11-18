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
Example: SPMD Execution Model with Direct MPI Collective Operations (Calls) in Torcalia

Description: This example demonstrates combining Torcalia's SPMD execution model with direct MPI collective operations (calls).

Workflow:
    1. The root node (rank 0) initializes array A with values [0, 100, 200, ...].
    2. A is broadcast to all nodes via MPI.Bcast!.
    3. Each node scales its copy of A by its rank (A .*= rank).
    4. The scaled arrays are summed element-wise across all nodes via MPI.Allreduce.

Requirements: 2+ MPI processes, 1+ worker thread(s) per process.

Julia < v1.12
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2 spmd_bcast_reduce.jl

Julia >= v1.12
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2,0 spmd_bcast_reduce.jl
"""

using Torcalia

using MPI

const N = 16
const A = zeros(Int, N)

function bcast_task()
    comm = MPI.COMM_WORLD
    MPI.Bcast!(A, comm)
end

function local_computation()
    node_id = Torcalia.node_id()
    
    A .*= node_id
    
    println("Node $node_id (after local computation): A = $A.")
end

function reduce_task()
    node_id = Torcalia.node_id()

    comm = MPI.COMM_WORLD
    result = MPI.Allreduce(A, +, comm)

    println("Node $node_id (after reduction): Element-wise sum of A = $result.")
end

function main()
    nodes = Torcalia.num_nodes()

    if nodes < 2
        println("[MPI Rank: $(Torcalia.node_id())] this example needs at least two MPI processes. exiting...")
        exit()
    end

    println("Running with $nodes MPI nodes.")

    for i in 1:N
        A[i] = 100 * (i - 1)
    end
    
    println("Node 0 (initialization): A = $A.")

    Torcalia.spmd(bcast_task)
    Torcalia.spmd(local_computation)
    Torcalia.spmd(reduce_task)
    
    println("All nodes completed their assigned tasks successfully.")
end

Torcalia.init(main)