import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

memory_list = np.array([])
# [proc.info for proc in psutil.process_iter(['pid', 'name']) if "python" in proc.info["name"]]
# pid = [proc.info for proc in psutil.process_iter(['pid', 'name']) if "python" in proc.info["name"]][0]["pid"]
pid = 20860
# print(pid)
t0 = perf_counter()
max_time = 10
while perf_counter() - t0 < max_time:
    memory_list = np.append(memory_list, psutil.Process(pid).memory_info().rss / 1024 ** 2)
print(memory_list.max())
plt.plot(memory_list)
plt.show()