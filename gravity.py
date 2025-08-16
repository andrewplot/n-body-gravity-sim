import numpy as np
import random
import time

def calcAccel():
    return

#runtime code
start = time.time()
for t in range(1000000):
    x = t**2

#constant decl
M = 1
dt = 0.05
g = 9.81
G = 6.674e-11
N = 10
x_min = y_min = -100
x_max = y_max = 100

#arr decl
mass = [M] * N
vel = [[0,0]] * N
pos = [[0,0]] * N
pos = np.array(pos) #needed to allow subtracting lists for r_ij


#pos decl
for i in range(N):
    pos[i] = [random.uniform(x_min, x_max), random.uniform(y_min, y_max)]


#acceleration computations via newton's law; sums individual accel contributions
accel = [[0,0]] * N

for i in range(N):
    for j in range(N):
        if i == j: #skips on itself
            continue

        r_ij = pos[j] - pos[i]
        r_mag = np.linalg.norm(r_ij)

        if r_mag == 0: #skips other coincident points
            continue

        a_ij = G * mass[j] * r_ij / (r_mag ** 3)
        accel[i] += a_ij

print(pos)
print(accel)

#rk4
k1_pos = vel
k1_vel = accel
k2_pos = [0] * N
k2_vel = [0] * N
k3_pos = [0] * N
k3_vel = [0] * N
k4_pos = [0] * N
k4_vel = [0] * N

for i in range(N):
    temp_pos = pos[i] + (dt/2) * k1_pos[i]
    k2_pos[i] = vel[i] + (dt/2) * k1_vel[i]
    k2_vel[i] = calcAccel(temp_pos) 

    temp_pos = pos[i] + (dt/2) * k2_pos[i]
    k3_pos[i] = vel[i] + (dt/2) * k2_vel[i]
    k3_vel[i] = calcAccel(temp_pos)

    temp_pos = pos[i] + dt * k3_pos[i]
    k4_pos[i] = vel + dt * k3_vel
    k4_vel[i] = calcAccel(temp_pos)
    
    pos[i] += (dt/6) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
    vel[i] += (dt/6) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

#ui



#maybe add leapfrog and have it overlay, depicting tradeoff between energy conservation and accuracy in long run







end = time.time()
print("Runtime:", end-start, "s")

        

        
        