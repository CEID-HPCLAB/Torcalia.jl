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
Example: Passing Positional and Keyword Arguments to Tasks in Torcalia

Description: This example demonstrates how to pass both positional and keyword arguments to tasks submitted to the Torcalia AMT runtime system. 

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 4 args.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 4,0 args.jl
"""

import Torcalia as Torc

function work(a::Int, b::Int; c::Int = 0, d::Int = 0)::Int
    sleep(0.1 + 0.05 * rand())
    
    result = a + b + c + d
    println("Worker on node $(Torc.node_id()): Computing $a + $b + $c + $d = $result.")
    
    result
end

function main()
    ntasks = 12

    tasks = Torc.Future[]
    
    for i in 1:4
        for j in 1:3
            pos_args = (i, j)
            kw_args = (; c = i * 10, d = j * 5)
            
            push!(tasks, Torc.submit(work, pos_args...; kw_args...))
        end
    end

   println("Submitted $ntasks tasks.")

    for (idx, t) in enumerate(tasks)
        pos = Torc.input(t)
        kw = Torc.kw_input(t)
        
        pos_str = join(pos, ", ")
        kw_str = join(["$k = $v" for (k, v) in pairs(kw)], ", ")
        
        println("Task $idx - Positional: ($pos_str) - Keyword: ($kw_str).")
    end

    Torc.wait(tasks)

    for (idx, t) in enumerate(tasks)
        pos = Torc.input(t)

        kw = Torc.kw_input(t)
        res = Torc.result(t)
        
        pos_str = join(pos, ", ")
        kw_str = join(["$k = $v" for (k, v) in pairs(kw)], ", ")
        
        println("Task $idx: work($pos_str; $kw_str) = $res.")
    end
    
    println("All tasks completed successfully.")
end

Torc.init(main)