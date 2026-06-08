#! /usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

colors = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#CC79A7']
color_total = colors[0]
color_solve = colors[1]
color_rhs = colors[2]
color_matrix = colors[3]
color_ideal = colors[4]

n_procs = [1, 2, 4, 8, 12]


total_time = [20681.9, 10468.9, 5262.7, 4326.9, 4689.3]
solve_time = [13742, 6483.8, 3084.8, 2309.3, 2264.9]
rhs_time = [1573.6, 819.8, 435.5, 323.4, 350.5]
matrix_time = [352.3, 240.8, 120.2, 87.4, 79.8]
average_throughput = [57.9, 114.3, 227.5, 276.7, 255.3]

def compute_ideal_time(time):
    return [time[0]/n_procs[0], time[0]/n_procs[1], time[0]/n_procs[2],
            time[0]/n_procs[3], time[0]/n_procs[4]]

ideal_strong_total_time = compute_ideal_time(total_time)
ideal_strong_solve_time = compute_ideal_time(solve_time)
ideal_strong_rhs_time = compute_ideal_time(rhs_time)
ideal_strong_matrix_time = compute_ideal_time(matrix_time)

def compute_speedup(time):
    return [time[0]/time[0], time[0]/time[1], time[0]/time[2], time[0]/time[3],
            time[0]/time[4]]

speedup_total = compute_speedup(total_time)
speedup_solve = compute_speedup(solve_time)
speedup_rhs = compute_speedup(rhs_time)
speedup_matrix = compute_speedup(matrix_time)


average_throughput_speedup = compute_speedup(average_throughput)

######################
# PLOT PARAMETERS
######################

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

######################
# TIME
######################
fig, ax = plt.subplots(figsize=(16, 12), dpi = 100)
total_time_plt, = plt.loglog(n_procs, total_time, '*-', color = color_total, linewidth = 3, markersize = 12,
        label='Total')
ideal_strong_total_time_plt, = plt.loglog(n_procs, ideal_strong_total_time, '--',
        color = color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Strong Scaling')
solve_time_plt, = plt.loglog(n_procs, solve_time, '*-', color = color_solve, linewidth = 3, markersize = 12,
        label='Solve')
ideal_strong_solve_time_plt, = plt.loglog(n_procs, ideal_strong_solve_time, '--', color
        = color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Strong Scaling')
rhs_time_plt, = plt.loglog(n_procs, rhs_time, '*-',  color = color_rhs, linewidth = 3, markersize = 12,
        label='RHS')
ideal_strong_rhs_time_plt, = plt.loglog(n_procs, ideal_strong_rhs_time, '--', color =
        color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Strong Scaling')
matrix_time_plt, = plt.loglog(n_procs, matrix_time, '*-',  color = color_matrix, linewidth = 3, markersize = 12,
        label='Matrix')
ideal_strong_matrix_time_plt, = plt.loglog(n_procs, ideal_strong_matrix_time, '--', color =
        color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Strong Scaling')
ax.legend(handles=[total_time_plt, solve_time_plt, rhs_time_plt,
    matrix_time_plt, ideal_strong_total_time_plt], fontsize = 18)
plt.xlabel('Number of cores', fontsize = 20)
plt.ylabel('Time (s)', fontsize = 20)
ax.grid(which='both')
plt.savefig('dwell_rook_mecha_time.png')
#plt.show()
plt.clf()

######################
# SPEEDUP
#####################
fig, ax = plt.subplots(figsize=(16, 12), dpi = 100)
speedup_total_plt, = plt.plot(n_procs, speedup_total, '-', color =
        color_total, linewidth = 3, markersize = 12,
        label='Total')
speedup_solve_plt, = plt.plot(n_procs, speedup_solve, '-', color =
        color_solve, linewidth = 3, markersize = 12,
        label='Solve')
speedup_rhs_plt, = plt.plot(n_procs, speedup_rhs, '-', color =
        color_rhs, linewidth = 3, markersize = 12,
        label='RHS')
speedup_matrix_plt, = plt.plot(n_procs, speedup_matrix, '-', color =
        color_matrix, linewidth = 3, markersize = 12,
        label='Matrix')
ideal_speedup_plt, = plt.plot(n_procs, n_procs , '-', color = color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Scaling')

plt.xlim(1, 12)
plt.ylim(1, 8)
plt.xticks(np.arange(1, 13, 1))
plt.yticks(np.arange(1, 9, 1))
ax.legend(handles=[speedup_total_plt, speedup_solve_plt, speedup_rhs_plt,
    speedup_matrix_plt, ideal_speedup_plt], fontsize = 18)
plt.xlabel('Number of cores', fontsize = 20)
plt.ylabel('Speedup', fontsize = 20)
ax.grid(which='both')
plt.savefig('dwell_rook_mecha_speedup.png')
#plt.show()
plt.clf()

######################
# BAR
######################
fig, ax = plt.subplots(figsize=(16, 12), dpi = 100)

time_left = [total_time[0] - solve_time[0] - rhs_time[0] - matrix_time[0],
             total_time[1] - solve_time[1] - rhs_time[1] - matrix_time[1],
             total_time[2] - solve_time[2] - rhs_time[2] - matrix_time[2],
             total_time[3] - solve_time[3] - rhs_time[3] - matrix_time[3],
             total_time[4] - solve_time[4] - rhs_time[4] - matrix_time[4]]

offset_1 = [solve_time[0]+rhs_time[0], solve_time[1]+rhs_time[1],
        solve_time[2]+rhs_time[2], solve_time[3]+rhs_time[3],
        solve_time[4]+rhs_time[4]]
offset_2 = [offset_1[0]+matrix_time[0], offset_1[1]+matrix_time[1],
        offset_1[2]+matrix_time[2], offset_1[3]+matrix_time[3],
        offset_1[4]+matrix_time[4]]

n_procs_str = ['1', '2', '4', '8', '12']
plt.bar(n_procs_str, solve_time, color=color_solve)
plt.bar(n_procs_str, rhs_time, bottom=solve_time, color=color_rhs)
plt.bar(n_procs_str, matrix_time, bottom=offset_1, color=color_matrix)
plt.bar(n_procs_str, time_left, bottom=offset_2, color=color_total)
plt.xlabel('Number of cores', fontsize = 20)
plt.ylabel('Time (s)', fontsize = 20)
plt.legend(["Solve", "RHS", "Matrix", "Total"])
plt.savefig('dwell_rook_mecha_bar.png')
#plt.show()
