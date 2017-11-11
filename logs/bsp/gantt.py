import matplotlib.pyplot as plt
import numpy as np


start_4 = np.loadtxt("node4_start_toload")
processing_4 = np.loadtxt("node4_processing_toload")
tuples_node4 = zip(start_4,processing_4)

start_5 = np.loadtxt("node5_start_toload")
processing_5 = np.loadtxt("node5_processing_toload")
tuples_node5 = zip(start_5,processing_5)

start_6 = np.loadtxt("node6_start_toload")
processing_6 = np.loadtxt("node6_processing_toload")
tuples_node6 = zip(start_6,processing_6)

start_9 = np.loadtxt("node9_start_toload")
processing_9 = np.loadtxt("node9_processing_toload")
tuples_node9 = zip(start_9,processing_9)

fig, ax = plt.subplots()
ax.broken_barh(tuples_node4[:8], (10, 9), facecolors=('sandybrown','mediumspringgreen','darkkhaki','olivedrab','seagreen','darkorchid','navy','gray'))


ax.broken_barh(tuples_node5[:8], (20, 9),facecolors=('sandybrown','mediumspringgreen','darkkhaki','olivedrab','seagreen','darkorchid','navy','gray'))

ax.broken_barh(tuples_node6[:8], (30, 9),facecolors=('sandybrown','mediumspringgreen','darkkhaki','olivedrab','seagreen','darkorchid','navy','gray'))

ax.broken_barh(tuples_node9[:8], (40, 9),facecolors=('sandybrown','mediumspringgreen','darkkhaki','olivedrab','seagreen','darkorchid','navy','gray'))


#ax.broken_barh([(10, 50), (100, 20), (130, 10)], (30, 9),
#               facecolors=('red', 'yellow', 'green'))
#ax.set_ylim(5, 35)
#ax.set_xlim(0, 200)
ax.set_xlabel('seconds since start')
ax.set_ylabel('Worker Nodes')
ax.set_yticks([15, 25, 35, 45])
ax.set_yticklabels(['Node4', 'Node5', 'Node6', 'Node9'])
ax.grid(True)

'''
ax.annotate('race interrupted', (61, 25),
            xytext=(0.8, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')
'''
plt.show()

