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
Example: Nested Parallelism in Torcalia (Parallel Recursive Fibonacci Computation)

Description: This example demonstrates Torcalia's support for nested parallelism through the recursive computation of the n-th Fibonacci number.  
             Multiple levels of parallelism are exposed by recursively submitting two tasks within each task function and waiting for their results, 
             forming a hierarchical dependency graph. To avoid task oversubscription, the depth of recursive task spawning is bounded. 

Note: Since this example is intended to show Torcalia's support for nested parallelism, identical function calls are recomputed rather than memoized. 

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3 nested_parallelism.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/julia/environment --threads 3,0 nested_parallelism.jl
"""

import Torcalia as Torc

function fibonacci(n::Int)::Int
    n < 2 && return n
    n < 30 && return fibonacci(n - 1) + fibonacci(n - 2)

    taskA = Torc.submit(fibonacci, n - 1)
    taskB = Torc.submit(fibonacci, n - 2)

    Torc.wait([taskA, taskB])
    Torc.result(taskA) + Torc.result(taskB)
end

function main()
    N = 45

    t0 = Torc.gettime()
   
    value = fibonacci(N)

    elapsed = Torc.gettime() - t0
   
    @assert value == 1_134_903_170
    
    println("Fibonacci($N) = $value.")
    println("All tasks completed successfully.")
    println("Total elapsed time: $(round(elapsed, digits = 3)) seconds.")
end

Torc.init(main)