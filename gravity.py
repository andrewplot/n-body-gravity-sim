import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time
# import keyboard #for exit condition; cant get to work

#constant declarations
G = 6.674e-11
N = 20
M = 10000000
dt = 30
g = 9.81
mass = [M] * N


x_min = y_min = -100
x_max = y_max = 100

def initialize_pos(pos: list):
    for i in range(N):
        pos[i] = [random.uniform(x_min, x_max), random.uniform(y_min, y_max)]
    return pos

def calc_accel(i: int, pos: list):
    acceleration = np.array([0.0, 0.0])

    for j in range(N):
        if i == j: #skips on itself
            continue

        r_ij = pos[j] - pos[i]
        r_mag = np.linalg.norm(r_ij)
        
        if r_mag == 0: #skips other coincident points
            continue
        a_ij = G * mass[j] * r_ij / (r_mag ** 3)
        acceleration += a_ij
    return acceleration

def runge_kutta4(pos, vel, accel):
    k1_pos = vel
    k1_vel = accel
    k2_pos = np.zeros((N, 2), dtype=float)
    k2_vel = np.zeros((N, 2), dtype=float)
    k3_pos = np.zeros((N, 2), dtype=float)
    k3_vel = np.zeros((N, 2), dtype=float)
    k4_pos = np.zeros((N, 2), dtype=float)
    k4_vel = np.zeros((N, 2), dtype=float)

    #k2
    temp_pos = pos + (dt/2) * k1_pos
    for i in range(N):
        k2_pos[i] = vel[i] + (dt/2) * k1_vel[i]
        k2_vel[i] = calc_accel(i, temp_pos)
    
    #k3    
    temp_pos = pos + (dt/2) * k2_pos
    for i in range(N):
        k3_pos[i] = vel[i] + (dt/2) * k2_vel[i]
        k3_vel[i] = calc_accel(i, pos)
    
    #k4
    temp_pos = pos + dt * k3_pos
    for i in range(N):
        k4_pos[i] = vel[i] + dt * k3_vel[i]
        k4_vel[i] = calc_accel(i, pos)

    #update    
    pos += (dt/6) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
    vel += (dt/6) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

    return pos, vel

def animate(frame, pos, vel, accel, scat):
    #calculate accelerations (newton's law, summing individual accel contributions)
    for i in range(N):
        accel[i] = calc_accel(i, pos)

    #rk4
    pos, vel = runge_kutta4(pos, vel, accel)

    #update
    scat.set_offsets(pos)

    return scat,


def main():
    #runtime start
    start = time.time()
    for t in range(1000000):
        x = t**2

    #array declarations
    vel = np.zeros((N,2), dtype=float)
    pos = np.zeros((N,2), dtype=float)
    pos = np.array(pos) #needed to allow subtracting lists for r_ij
    accel = np.zeros((N,2), dtype=float)

    #initialize coords and plot
    pos = initialize_pos(pos)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(x_min*3, x_max*3)
    ax.set_ylim(y_min*3, y_max*3)
    ax.set_aspect('equal')
    scat = ax.scatter(pos[:, 0], pos[:, 1], s=2, c='blue')

    #animation
    animation = FuncAnimation(fig, animate, fargs=(pos, vel, accel, scat), frames=100, interval=50, blit=False)
    plt.show()

    # runtime end
    end = time.time()
    print("Runtime:", end-start, "s")
    return 0

if __name__ == "__main__":
    main()

#maybe add leapfrog and have it overlay, depicting tradeoff between energy conservation and accuracy in long run; other options: rkn4, forest-ruth, etc.
#optimizations: gpu (mac: pytorch with MPS), fast multipole methods, barnes-hut