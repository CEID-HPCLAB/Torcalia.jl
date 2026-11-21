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

import Torcalia as Torc

square(x::Int)::Int = x^2

double(x::Torc.Future)::Int = 2 * Torc.result(x)

function add(x::Torc.Future, y::Torc.Future)::Int
    Torc.result(x) + Torc.result(y)
end

function main()
    taskA = Torc.submit(square, 2) # 4
    taskB = Torc.submit(square, 4) # 16

    taskC = Torc.submit(add, taskA, taskB) # 4 + 16 = 20
    taskD = Torc.submit(double, taskA) # 2 * 4 = 8

    Torc.wait([taskC, taskD])

    @assert Torc.result(taskC) == 20
    @assert Torc.result(taskD) == 8
end

Torc.init(main)