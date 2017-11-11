import matplotlib.pyplot as plt
import numpy as np
y = np.loadtxt("processed_more_local_0.05")
z = np.loadtxt("processed_more_local_incr")
w = np.loadtxt("processed_more_local_decr")
line_up, = plt.plot(y, label="constant rate")
line_down, = plt.plot(z, label="increasing rate")
decr_rate, = plt.plot(w, label="decreasing rate")
plt.legend(handles=[line_up, line_down,decr_rate])
plt.show()

