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
Example: Distributed Shared Memory (DSM) with ArgoDSM in Torcalia

Description: This example demonstrates how the DSM abstraction can be used on top of Torcalia. Users only need to 
             explicitly allocate the corresponding shared memory regions through the `pdsm_alloc` function. The runtime 
             transparently manages the underlying communication and synchronization using the primitives provided by ArgoDSM, 
             allowing the allocated data to be accessed within tasks as in shared-memory programming.

Notes: In this example, the primary application task allocates and initializes the shared memory region `A`, which is 
       accessible to tasks running on different MPI nodes. Then, `N` tasks are created, where `N` is the number of MPI nodes 
       used to launch Torcalia. Each task is explicitly scheduled to a different MPI node and executes the `work` function, 
       which writes the MPI rank of its assigned node to the corresponding position in `A`.

Requirements: 2+ MPI processes, 1+ worker thread(s) per process.

Julia < v1.12:
Run: mpiexecjl -n 8 julia --project=/path/to/julia/environment --threads 2 dsm.jl

Julia >= v1.12:
Run: mpiexecjl -n 8 julia --project=/path/to/julia/environment --threads 2,0 dsm.jl
"""

import Torcalia as Torc

function work(A::Vector{Int})
    rank = Torc.node_id(); A[rank + 1] = rank
    Torc.argo_barrier(1)
end

function main()
    N = Torc.num_nodes()
    
    A = Torc.pdsm_alloc(Int(N), Int)

    println("[MPI Rank: $(Torc.node_id())] Allocated ArgoDSM array A ($N integer elements).")
    
    for i in 1:N A[i] = 1 end

    println("[MPI Rank: $(Torc.node_id())] Initial values of A: $A.")
    
    tasks = Vector{Torc.Future}(undef, N)
    
    for i in 0:N-1
        tasks[i+1] = Torc.submit(work, A; node = i)
    end
    
    Torc.wait(tasks)

    println("[MPI Rank: $(Torc.node_id())] Final values of A after all tasks completed: $A.")

    @assert A == collect(Int, 0:N-1)
end

# Configure the ArgoDSM memory and cache capacities
Torc.argo_init(1024*1024, 1024*4)
Torc.init(main)