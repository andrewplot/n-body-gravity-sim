import numpy as np
import random
import time

#constant declarations
G = 6.674e-11
N = 10
M = 1
dt = 0.05
g = 9.81

x_min = y_min = -100
x_max = y_max = 100

def initialize_pos(pos: list, x_min: int, x_max: int, y_min: int, y_max: int):
    for i in range(N):
        pos[i] = [random.uniform(x_min, x_max), random.uniform(y_min, y_max)]
    return pos

def calc_accel(i, pos, mass):
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
    k2_pos = np.zeros(N)
    k2_vel = np.zeros(N)
    k3_pos = np.zeros(N)
    k3_vel = np.zeros(N)
    k4_pos = np.zeros(N)
    k4_vel = np.zeros(N)

    for i in range(N):
        temp_pos = pos[i] + (dt/2) * k1_pos[i]
        k2_pos[i] = vel[i] + (dt/2) * k1_vel[i]
        k2_vel[i] = calc_accel(temp_pos) 
        
        temp_pos = pos[i] + (dt/2) * k2_pos[i]
        k3_pos[i] = vel[i] + (dt/2) * k2_vel[i]
        k3_vel[i] = calc_accel(temp_pos)

        temp_pos = pos[i] + dt * k3_pos[i]
        k4_pos[i] = vel + dt * k3_vel
        k4_vel[i] = calc_accel(temp_pos)
        
        pos[i] += (dt/6) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        vel[i] += (dt/6) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

    return pos, vel



def main():
    #runtime start
    start = time.time()
    for t in range(1000000):
        x = t**2

    #array declarations
    mass = [M] * N
    vel = [[0,0]] * N
    pos = [[0,0]] * N
    pos = np.array(pos) #needed to allow subtracting lists for r_ij
    accel = [[0,0]] * N

    pos = initialize_pos(pos, x_min, x_max, y_min, y_max)

    while True:
        #calculate accelerations (newton's law, summing individual accel contributions)
        for i in range(N):
            accel[i] = calc_accel(i, pos, mass)

        #rk4
        pos, vel = runge_kutta4(pos, vel, accel)

        #ui
        print(pos)
        print(vel)
        print(accel)
        break


#maybe add leapfrog and have it overlay, depicting tradeoff between energy conservation and accuracy in long run

    #runtime end
    end = time.time()
    print("Runtime:", end-start, "s")
    return 0

if __name__ == "__main__":
    main()
