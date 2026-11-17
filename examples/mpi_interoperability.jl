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
Example: MPI Interoperability in Torcalia

Description: This example demonstrates the interoperability of Torcalia with MPI. It showcases how developers can seamlessly 
             integrate the AMT parallelism model with direct MPI (collective) operations.

The example includes:
- MPI operations BEFORE Torcalia's runtime engine initialization (MPI.gather, MPI.Allreduce).
- MPI operations INSIDE Torcalia tasks (same operations as outside, but within the context of Torcalia tasks).
- multiple Torcalia.init() calls with MPI_finalize control.

Notes:
- `MPI_finalize = false`: Keeps MPI alive between Torcalia sessions.
- Direct MPI operations coexist with Torcalia task-based execution.

Requirements: 2+ MPI processes, 1+ worker thread(s) per process.

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2 mpi_interoperability.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 2,0 mpi_interoperability.jl
"""

using Torcalia

nodes = Torcalia.num_nodes()

if nodes < 2
    println("[Node: $(Torcalia.node_id())] this example needs at least two MPI processes. exiting...")
    exit()
end

using MPI

rank = MPI.Comm_rank(MPI.COMM_WORLD)   

function mpi_gather_outside()
    local_val = [rank + 1]
    all_vals = MPI.gather(local_val, MPI.COMM_WORLD)
    
    if rank == 0
        println("Outside Torcalia (gather) $(vcat(all_vals...)).")
    end
end

function mpi_gather_inside()
    user_comm = MPI.COMM_WORLD
    local_val = [rank + 1]
    all_vals = MPI.gather(local_val, user_comm)

    if rank == 0
        println("Inside Torcalia (gather) $(vcat(all_vals...)).")
    end
end

function torcalia_gather()
    Torcalia.spmd(mpi_gather_inside)
end

function mpi_allreduce_outside()
    local_val = rank + 1
    total_sum = MPI.Allreduce(local_val, +, MPI.COMM_WORLD)
    
    if rank == 0
        println("Outside Torcalia (Allreduce) $total_sum.")
    end
end

function mpi_allreduce_inside()
    user_comm = MPI.COMM_WORLD
    local_val = rank + 1
    total_sum = MPI.Allreduce(local_val, +, user_comm)
    
    if rank == 0
        println("Inside Torcalia (Allreduce) $total_sum.")
    end
end

function torcalia_allreduce()
    Torcalia.spmd(mpi_allreduce_inside)
end

mpi_gather_outside()

Torcalia.init(torcalia_gather; MPI_finalize = false)

mpi_allreduce_outside()

Torcalia.init(torcalia_allreduce)