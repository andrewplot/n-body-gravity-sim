import numpy as np
import random
import time

#runtime code
start = time.time()
for t in range(1000000):
    x = t**2

#constant decl
M = 1
Dt = 0.01
g = 9.81
G = 6.674e-11
N = 1000
x_min = y_min = -100
x_max = y_max = 100

#arr decl
mass = [M] * N
vel = [0] * N
pos = [[0,0]] * N

#pos decl
for i in range(N):
    pos[i] = [random.uniform(x_min, x_max), random.uniform(y_min, y_max)]
pos = np.array(pos) #needed to allow subtracting lists for r_ij


#acceleration computations via newton's law; sums individual accel contributions
accel = [[0,0]] * N

for i in range(N):
    for j in range(N):
        r_ij = pos[j] - pos[i]
        r_mag = np.linalg.norm(r_ij)

        if r_mag == 0: #skips on itself and other coincident points
            continue

        a_ij = G * mass[j] * r_ij / (r_mag ** 3)
        accel[i] += a_ij

print(pos)
print(accel)

#rk4 step









end = time.time()
print("Runtime:", end-start, "s")

        

        
        