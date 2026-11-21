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
Example: Task Prioritization in Torcalia

Description: This example demonstrates Torcalia's priority-based task scheduling. Tasks are scheduled according to their priority level, 
             with higher-priority tasks executing before lower-priority ones when workers become available.

Priority levels:
- `TaskPriority.high`: urgent tasks (green output)
- `TaskPriority.medium`: medium priority tasks (yellow output)
- `TaskPriority.low`: low priority tasks (cyan output, default)

Workflow:
- 8 low-priority tasks are submitted first.
- 8 medium-priority tasks are submitted second.
- 8 high-priority tasks are submitted last.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2 priority.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/julia/environment --threads 2,0 priority.jl
"""

import Torcalia as Torc

using Crayons

const PRIORITY_COLOR = Dict(
    :low    => :green,
    :medium => :yellow,
    :high   => :red
)

function work(i::Int, p::Torc.TaskPriorityLevel)
    sleep(0.5)
    
    cr = Crayon(foreground = PRIORITY_COLOR[p.level])
    println(cr("Task $i finished (priority = $(p.level))."))
end

function main()
    println(Crayon(foreground = :magenta)("Submitting 24 tasks: 8 low, 8 medium, 8 high priority."))
    
    priorities = vcat(
        fill(Torc.TaskPriority.low, 8),
        fill(Torc.TaskPriority.medium, 8),
        fill(Torc.TaskPriority.high, 8)
    )

    tasks = Torc.Future[]

    for (i, p) in enumerate(priorities)
        push!(tasks, Torc.submit(work, i, p; priority = p))
    end

    Torc.wait(tasks)
    
    println(Crayon(foreground = :magenta)("All tasks completed successfully."))
end

Torc.init(main)