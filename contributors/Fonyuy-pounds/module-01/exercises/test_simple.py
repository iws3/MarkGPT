import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("Numpy version:", np.__version__)
print("Matplotlib version:", matplotlib.__version__)

# Simple plot
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig("test_plot.png")
print("Saved test_plot.png")
