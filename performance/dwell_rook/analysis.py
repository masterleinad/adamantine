#! /usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

colors = ['#E69F00', '#56B4E9', '#009E73', '#0072b2', '#CC79A7']
color_coarse = colors[0]
color_fine = colors[2]
color_p2 = colors[1]
color_p3 = colors[3]
color_ideal = colors[4]

n_procs = [1, 2, 4, 8, 12]

# Coarse Mesh
#------------
time_coarse = [6031.9, 4320.7, 3020.7, 3026.3, 3252.9]
time_coarse_0 = time_coarse[0]
ideal_strong_time_coarse = [time_coarse_0, time_coarse_0/2, time_coarse_0/4,
        time_coarse_0/8, time_coarse_0/12]

speedup_coarse = [time_coarse_0/x for x in time_coarse]
efficiency_coarse = [time_coarse_0/time_coarse_0, time_coarse_0/(time_coarse[1]*2),
        time_coarse_0/(time_coarse[2]*4), time_coarse_0/(time_coarse[3]*8),
        time_coarse_0/(time_coarse[4]*12)]

average_throughput_coarse = [1840.1, 2568.8, 3674.4, 3667.6, 3412.1]
realtime_throughput_coarse = [1000, 1000, 1000, 1000, 1000]
first_block_throughput_coarse = [3106.7, 4407.5, 4498.6, 4067.8, 3537.2]
last_block_throughput_coarse = [1314.9, 1996.9, 3100.4, 3350.6, 3228.2]
average_throughput_coarse_0 = average_throughput_coarse[0]
average_throughput_speedup_coarse = [x/average_throughput_coarse_0 for x in average_throughput_coarse]
first_throughput_coarse_0 = first_block_throughput_coarse[0]
first_block_throughput_speedup_coarse = [x/first_throughput_coarse_0 for x in first_block_throughput_coarse]
last_throughput_coarse_0 = last_block_throughput_coarse[0]
last_block_throughput_speedup_coarse = [x/last_throughput_coarse_0 for x in last_block_throughput_coarse]

eval_percent_coarse = [75.8, 66.6, 69.6, 60.9, 58.4]

# Fine Mesh
#----------
time_fine = [27993.6, 15656.3, 10166.0, 6267.9, 5691.6]
time_fine_0 = time_fine[0]
ideal_strong_time_fine = [time_fine_0, time_fine_0/2, time_fine_0/4,
        time_fine_0/8, time_fine_0/12]

speedup_fine = [time_fine_0/x for x in time_fine]
efficiency_fine = [time_fine_0/time_fine_0, time_fine_0/(time_fine[1]*2),
        time_fine_0/(time_fine[2]*4), time_fine_0/(time_fine[3]*8),
        time_fine_0/(time_fine[4]*12)]

average_throughput_fine = [396.5, 708.9, 1091.8, 1770.8, 1950.1]
first_block_throughput_fine = [1188.5, 1752.5, 2247.6, 2525.9, 2497.7]
last_block_throughput_fine = [239.0, 444.9, 733.0, 1327.0, 1530.8]
average_throughput_fine_0 = average_throughput_fine[0]
average_throughput_speedup_fine = [x/average_throughput_fine_0 for x in average_throughput_fine]
first_throughput_fine_0 = first_block_throughput_fine[0]
first_block_throughput_speedup_fine = [x/first_throughput_fine_0 for x in first_block_throughput_fine]
last_throughput_fine_0 = last_block_throughput_fine[0]
last_block_throughput_speedup_fine = [x/last_throughput_fine_0 for x in last_block_throughput_fine]

eval_percent_fine = [95.3, 90.7, 86.7, 77.5, 72.1]

# P2-Refinement
#--------------
time_p2 = [12682.6, 7836.5, 4757.7, 4037.8, 3927.9]
time_p2_0 = time_p2[0]
ideal_strong_time_p2 = [time_p2_0, time_p2_0/2, time_p2_0/4,
        time_p2_0/8, time_p2_0/12]

speedup_p2 = [time_p2_0/x for x in time_p2]
efficiency_p2 = [time_p2_0/time_p2_0, time_p2_0/(time_p2[1]*2),
        time_p2_0/(time_p2[2]*4), time_p2_0/(time_p2[3]*8),
        time_p2_0/(time_p2[4]*12)]

average_throughput_p2 = [875.1, 1415.3, 2332.9, 2748.8, 2825.7]
first_block_throughput_p2 = [2097.4, 2682.0, 3870.1, 3602.9, 3326.9]
last_block_throughput_p2 = [564.3, 978.2, 1690.4, 2230.8, 2427.2]
average_throughput_p2_0 = average_throughput_p2[0]
average_throughput_speedup_p2 = [x/average_throughput_p2_0 for x in average_throughput_p2]
first_throughput_p2_0 = first_block_throughput_p2[0]
first_block_throughput_speedup_p2 = [x/first_throughput_p2_0 for x in first_block_throughput_p2]
last_throughput_p2_0 = last_block_throughput_p2[0]
last_block_throughput_speedup_p2 = [x/last_throughput_p2_0 for x in last_block_throughput_p2]

eval_percent_fine = [87.8, 79.7, 79.2, 68.7, 64.2]

# P3-Refinement
#--------------
time_p3 = [29076.8, 15634.3, 8973.7, 6293.8, 5507.1]
time_p3_0 = time_p3[0]
ideal_strong_time_p3 = [time_p3_0, time_p3_0/2, time_p3_0/4,
        time_p3_0/8, time_p3_0/12]

speedup_p3 = [time_p3_0/x for x in time_p3]
efficiency_p3 = [time_p3_0/time_p3_0, time_p3_0/(time_p3[1]*2),
        time_p3_0/(time_p3[2]*4), time_p3_0/(time_p3[3]*8),
        time_p3_0/(time_p3[4]*12)]

average_throughput_p3 = [381.7, 709.9, 1236.9, 1763.5, 2015.4]
first_block_throughput_p3 = [1257.6, 1847.4, 2822.3, 2970.6, 2886.9]
last_block_throughput_p3 = [203.3, 435.2, 769.8, 1251.8, 1528.3]
average_throughput_p3_0 = average_throughput_p3[0]
average_throughput_speedup_p3 = [x/average_throughput_p3_0 for x in average_throughput_p3]
first_throughput_p3_0 = first_block_throughput_p3[0]
first_block_throughput_speedup_p3 = [x/first_throughput_p3_0 for x in first_block_throughput_p3]
last_throughput_p3_0 = last_block_throughput_p3[0]
last_block_throughput_speedup_p3 = [x/last_throughput_p3_0 for x in last_block_throughput_p3]

eval_percent_fine = [94.0, 91.9, 86.8, 76.5, 71.6]

######################
# PLOT PARAMETERS
######################

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

######################
# TIME
######################
fig, ax = plt.subplots(figsize=(16, 12), dpi = 100)
time_coarse_plt, = plt.loglog(n_procs, time_coarse, '*-', color = color_coarse, linewidth = 3, markersize = 12,
        label='Coarse Simulation')
ideal_strong_coarse_plt, = plt.loglog(n_procs, ideal_strong_time_coarse, '--',
        color = color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Strong Scaling')
time_fine_plt, = plt.loglog(n_procs, time_fine, '*-', color = color_fine, linewidth = 3, markersize = 12,
        label='h-refined Simulation')
ideal_strong_fine_plt, = plt.loglog(n_procs, ideal_strong_time_fine, '--', color
        = color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Strong Scaling')
time_p2_plt, = plt.loglog(n_procs, time_p2, '*-',  color = color_p2, linewidth = 3, markersize = 12,
        label='p2-refine Simulation')
ideal_strong_p2_plt, = plt.loglog(n_procs, ideal_strong_time_p2, '--', color =
        color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Strong Scaling')
time_p3_plt, = plt.loglog(n_procs, time_p3, '*-',  color = color_p3, linewidth = 3, markersize = 12,
        label='p3-refine Simulation')
ideal_strong_p3_plt, = plt.loglog(n_procs, ideal_strong_time_p3, '--', color =
        color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Strong Scaling')
ax.legend(handles=[time_coarse_plt, time_fine_plt, time_p2_plt, time_p3_plt, ideal_strong_coarse_plt], fontsize = 18)
plt.xlabel('Number of cores', fontsize = 20)
plt.ylabel('Time (s)', fontsize = 20)
ax.grid(which='both')
plt.savefig('dwell_rook_time.png')
#plt.show()
plt.clf()

######################
# SPEEDUP
#####################
fig, ax = plt.subplots(figsize=(16, 12), dpi = 100)
speedup_coarse_plt, = plt.plot(n_procs, speedup_coarse, '-', color =
        color_coarse, linewidth = 3, markersize = 12,
        label='Coarse Simulation')
speedup_fine_plt, = plt.plot(n_procs, speedup_fine, '-', color =
        color_fine, linewidth = 3, markersize = 12,
        label='h-refined Simulation')
speedup_p2_plt, = plt.plot(n_procs, speedup_p2, '-', color =
        color_p2, linewidth = 3, markersize = 12,
        label='p2-refined Simulation')
speedup_p3_plt, = plt.plot(n_procs, speedup_p3, '-', color =
        color_p3, linewidth = 3, markersize = 12,
        label='p3-refined Simulation')
ideal_speedup_plt, = plt.plot(n_procs, n_procs , '-', color = color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Scaling')
plt.xlim(1, 12)
plt.ylim(1, 8)
plt.xticks(np.arange(1, 13, 1))
plt.yticks(np.arange(1, 9, 1))
ax.legend(handles=[speedup_coarse_plt, speedup_fine_plt, speedup_p2_plt, speedup_p3_plt, ideal_speedup_plt], fontsize = 18)
plt.xlabel('Number of cores', fontsize = 20)
plt.ylabel('Speedup', fontsize = 20)
ax.grid(which='both')
plt.savefig('dwell_rook_speedup.png')
#plt.show()
plt.clf()

######################
# THROUGHPUT
######################
fig, ax = plt.subplots(figsize=(16, 12), dpi = 100)

avg_coarse_plt, = plt.plot(n_procs, average_throughput_coarse, '-', color =
        color_coarse, linewidth = 3, markersize = 12,
        label='Coarse: Average')
first_coarse_plt, = plt.plot(n_procs, first_block_throughput_coarse, '-.', color
        = color_coarse, linewidth = 3, 
        markersize = 12, label='Coarse: First Block')
last_coarse_plt, = plt.plot(n_procs, last_block_throughput_coarse, '--', color =
        color_coarse, linewidth = 3, 
        markersize = 12, label='Coarse: Last Block')

avg_fine_plt, = plt.plot(n_procs, average_throughput_fine, '-', color =
        color_fine, linewidth = 3, markersize = 12,
        label='h-refined: Average')
first_fine_plt, = plt.plot(n_procs, first_block_throughput_fine, '-.', color =
        color_fine, linewidth = 3, 
        markersize = 12, label='h-refined: First Block')
last_fine_plt, = plt.plot(n_procs, last_block_throughput_fine, '--', color =
        color_fine, linewidth = 3, 
        markersize = 12, label='h-refined: Last Block')

avg_p2_plt, = plt.plot(n_procs, average_throughput_p2, '-', color =
        color_p2, linewidth = 3, markersize = 12,
        label='p2-refined: Average')
first_p2_plt, = plt.plot(n_procs, first_block_throughput_p2, '-.',
        color = color_p2, linewidth = 3, 
        markersize = 12, label='p2-refined: First Block')
last_p2_plt, = plt.plot(n_procs, last_block_throughput_p2, '--', color
        = color_p2, linewidth = 3, 
        markersize = 12, label='p2-refined: Last Block')

avg_p3_plt, = plt.plot(n_procs, average_throughput_p3, '-', color =
        color_p3, linewidth = 3, markersize = 12,
        label='p-refined: Average')
first_p3_plt, = plt.plot(n_procs, first_block_throughput_p3, '-.',
        color = color_p3, linewidth = 3, 
        markersize = 12, label='p3-refined: First Block')
last_p3_plt, = plt.plot(n_procs, last_block_throughput_p3, '--', color
        = color_p3, linewidth = 3, 
        markersize = 12, label='p3-refined: Last Block')

realtime_coarse_plt, = plt.plot(n_procs, realtime_throughput_coarse, ':', color
        = color_ideal, linewidth = 3, markersize = 12, label='Real Time')
ax.legend(handles=[first_coarse_plt, avg_coarse_plt, last_coarse_plt,
    first_fine_plt, avg_fine_plt, last_fine_plt, first_p2_plt, avg_p2_plt,
    last_p2_plt, first_p3_plt, avg_p3_plt, last_p3_plt, realtime_coarse_plt], ncols = 3, fontsize = 18)
#plt.title('Throughput' , fontsize = 20)
plt.xlabel('Number of cores', fontsize = 20)
plt.ylabel('Iterations/s', fontsize = 20)
plt.xlim(1, 12)
plt.ylim(0, 5900)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(500))
ax.grid(True)
plt.savefig('dwell_rook_throughput.png')
#plt.show()
plt.clf()

######################
# THROUGHPUT SPEEDUP
######################
fig, ax = plt.subplots(figsize=(16, 12), dpi = 100)

avg_speedup_coarse_plt, = plt.plot(n_procs, average_throughput_speedup_coarse, '-', color =
        color_coarse, linewidth = 3, markersize = 12,
        label='Coarse: Average')
first_speedup_coarse_plt, = plt.plot(n_procs, first_block_throughput_speedup_coarse, '-.', color
        = color_coarse, linewidth = 3, 
        markersize = 12, label='Coarse: First Block')
last_speedup_coarse_plt, = plt.plot(n_procs, last_block_throughput_speedup_coarse, '--', color =
        color_coarse, linewidth = 3, 
        markersize = 12, label='Coarse: Last Block')

avg_speedup_fine_plt, = plt.plot(n_procs, average_throughput_speedup_fine, '-', color =
        color_fine, linewidth = 3, markersize = 12,
        label='h-refined: Average')
first_speedup_fine_plt, = plt.plot(n_procs, first_block_throughput_speedup_fine, '-.', color =
        color_fine, linewidth = 3, 
        markersize = 12, label='h-refined: First Block')
last_speedup_fine_plt, = plt.plot(n_procs, last_block_throughput_speedup_fine, '--', color =
        color_fine, linewidth = 3, 
        markersize = 12, label='h-refined: Last Block')

avg_speedup_p2_plt, = plt.plot(n_procs, average_throughput_speedup_p2, '-', color =
        color_p2, linewidth = 3, markersize = 12,
        label='Average: P2-refine')
first_speedup_p2_plt, = plt.plot(n_procs, first_block_throughput_speedup_p2, '-.',
        color = color_p2, linewidth = 3, 
        markersize = 12, label='p2-refined: First Block')
last_speedup_p2_plt, = plt.plot(n_procs, last_block_throughput_speedup_p2, '--', color
        = color_p2, linewidth = 3, 
        markersize = 12, label='p2-refined: Last Block')

avg_speedup_p3_plt, = plt.plot(n_procs, average_throughput_speedup_p3, '-', color =
        color_p3, linewidth = 3, markersize = 12,
        label='Average: P3-refine')
first_speedup_p3_plt, = plt.plot(n_procs, first_block_throughput_speedup_p3, '-.',
        color = color_p3, linewidth = 3, 
        markersize = 12, label='p3-refined: First Block')
last_speedup_p3_plt, = plt.plot(n_procs, last_block_throughput_speedup_p3, '--', color
        = color_p3, linewidth = 3, 
        markersize = 12, label='p3-refined: Last Block')
ideal_speedup_plt, = plt.plot(n_procs, n_procs , '-', color = color_ideal, linewidth = 3, markersize = 12,
        label='Ideal Scaling')

plt.xlim(1, 12)
plt.ylim(1, 8.8)
plt.xticks(np.arange(1, 13, 1))
plt.yticks(np.arange(1, 9, 1))
ax.legend(handles=[first_speedup_coarse_plt, avg_speedup_coarse_plt, last_speedup_coarse_plt,
    first_speedup_fine_plt, avg_speedup_fine_plt, last_speedup_fine_plt, first_speedup_p2_plt, avg_speedup_p2_plt,
    last_speedup_p2_plt, first_speedup_p3_plt, avg_speedup_p3_plt, last_speedup_p3_plt, ideal_speedup_plt],  ncols = 2, fontsize = 18)
#plt.title('Throughput' , fontsize = 20)
plt.xlabel('Number of cores', fontsize = 20)
plt.ylabel('Speedup', fontsize = 20)
plt.xlim(1, 12)
#plt.ylim(0, 4800)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(True)
plt.savefig('dwell_rook_throughput_speedup.png')
#plt.show()
plt.clf()
