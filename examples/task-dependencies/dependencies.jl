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
Example: Task Dependencies in Torcalia

Description: This example demonstrates how data dependencies between tasks are expressed in Torcalia, enabling dependency-aware dynamic 
             task scheduling.

Notes: In this example, Task C depends on the results of Tasks A and B and is therefore scheduled only after both complete, whereas Task D 
       depends only on Task A and can start as soon as A completes. Tasks A and B can be executed concurrently, as can Tasks C and D, 
       since they are independent of each other. 
       
Generated Task Dependency Directed Acyclic Graph (DAG):

    A = square(2) = 4         B = square(4) = 16
            │  ╲                      │
            │   ╲                     │
            │    └──────────┐         │
            │               │         │
            ▼               ▼         ▼
            D = double(A) = 8    C = add(A, B) = 20

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2 dependencies.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2,0 dependencies.jl
"""

using Torcalia

square(x::Int)::Int = x^2

double(x::Torcalia.Future)::Int = 2 * Torcalia.result(x)

function add(x::Torcalia.Future, y::Torcalia.Future)::Int
    Torcalia.result(x) + Torcalia.result(y)
end

function main()
    taskA = Torcalia.submit(square, 2) # 4
    taskB = Torcalia.submit(square, 4) # 16

    taskC = Torcalia.submit(add, taskA, taskB) # 4 + 16 = 20
    taskD = Torcalia.submit(double, taskA) # 2 * 4 = 8

    Torcalia.wait([taskC, taskD])

    @assert Torcalia.result(taskC) == 20
    @assert Torcalia.result(taskD) == 8
end

Torcalia.init(main)