import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time
# import keyboard #for exit condition; cant get to work

#constant declarations
G = 6.674e-11
N = 50
M = 10000000
dt = 0.1
g = 9.81
epsilon = 1e-1 #helps with not approaching infinitygit
mass = np.full(N, M)

x_min = y_min = -100
x_max = y_max = 100

def get_user_input():
    while True:
        user_input = input("1 for random, 2 for plummer sphere: ")

        try:
            user_input = int(user_input)
            if user_input in [1,2]:
                return user_input
            else:
                print("Err: Invalid input")
        except ValueError:
            print("Err: NaN input")


def randomize_pos(pos: list):
    for i in range(N):
        pos[i] = [random.uniform(x_min, x_max), random.uniform(y_min, y_max)]
    return pos

def plummer_sphere_pos(pos: list, vel: list):
    a = 10.0  # scale radius (characteristic size of the cluster)
    total_mass = N * M  # total mass of the system
    
    #pos = plummer sphere density profile
    for i in range(N):
        # Generate random radius using inverse transform sampling
        # For Plummer sphere: M(r) = M_total * r^3 / (r^2 + a^2)^(3/2)
        # Solving for r given a random cumulative mass fraction
        xi = random.random()  # random number between 0 and 1
        r = a / np.sqrt((xi**(-2/3) - 1))
        
        # Generate random angles for spherical coordinates
        theta = np.arccos(2 * random.random() - 1)  # uniform in cos(theta)
        phi = 2 * np.pi * random.random()  # uniform in phi
        
        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        
        pos[i] = [x, y]
    
    #vel = virial theorem and escape vel
    for i in range(N):
        r_i = np.linalg.norm(pos[i])
        
        # Calculate enclosed mass at radius r_i
        M_enc = total_mass * (r_i**3) / (r_i**2 + a**2)**(3/2)
        
        # Escape velocity at this radius
        v_esc = np.sqrt(2 * G * M_enc / r_i) if r_i > 0 else 0
        
        # Velocity magnitude (typically fraction of escape velocity for bound system)
        # Using King's approximation for velocity distribution
        v_mag = v_esc * random.uniform(0.1, 0.7)  # bound velocities
        
        # Random velocity direction
        v_theta = 2 * np.pi * random.random()
        v_x = v_mag * np.cos(v_theta)
        v_y = v_mag * np.sin(v_theta)
        
        vel[i] = [v_x, v_y]
    

    
    # Center of mass correction (ensure system doesn't drift)
    com_pos = np.mean(pos, axis=0)
    com_vel = np.mean(vel, axis=0)
    
    for i in range(N):
        pos[i] = pos[i] - com_pos
        vel[i] = vel[i] - com_vel
    
    return pos, vel

def total_energy(mass, vel, pos):
    #kinetic
    kinetic = 0
    for i in range(N):
        kinetic += 0.5 * mass[i] * (vel[i][0]**2 + vel[i][1]**2)

    #potential
    potential = 0
    for i in range(N):
        for j in range(i+1, N):
            r_ij = np.linalg.norm(pos[i] - pos[j])
            potential -= G * mass[i] * mass[j] / r_ij
    
    return kinetic + potential

def calc_accel(i: int, pos: list):
    acceleration = np.array([0.0, 0.0])

    for j in range(N):
        if i == j: #skips on itself
            continue

        r_ij = pos[j] - pos[i]
        r_mag = np.linalg.norm(r_ij)
        
        if r_mag == 0: #skips other coincident points
            continue
        #a_ij = G * mass[j] * r_ij / (r_mag ** 3)                       #correct formula
        a_ij = G * mass[j] * r_ij / ((r_mag**2 + epsilon**2) ** 1.5)    #formula w epsilon to prevent diverging force
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

def leapfrog(pos, vel, accel):
    vel += 0.5 * dt * accel
    pos += dt * vel
    
    new_accel = np.zeros((N, 2), dtype=float)
    for i in range(N):
        new_accel[i] = calc_accel(i, pos)
    
    vel += 0.5 * dt * new_accel

    return pos, vel

def animate(frame, pos, vel, accel, scat, start, energy_data, energy_line, ax_energy):
    #calc accelerations (newton's law, summing individual accel contributions)
    for i in range(N):
        accel[i] = calc_accel(i, pos)

    #numerical integration
    pos, vel = runge_kutta4(pos, vel, accel)
    #pos, vel = leapfrog(pos, vel, accel)

    #energy calc
    energy = total_energy(mass, vel, pos)
    energy_data.append(energy)

    #update
    scat.set_offsets(pos)
    energy_line.set_data(range(len(energy_data)), energy_data)
    ax_energy.relim()
    ax_energy.autoscale_view()
    
    #frametime
    end = time.time()
    print(f"Frametime: {end-start[0]:.18f} s")
    start[0] = time.time()

    return scat, energy_line


def main():
    #runtime start
    start = [time.time(), time.time()] #[frame_time, run_time]

    for t in range(1000000):
        x = t**2

    #array declarations
    vel = np.zeros((N,2), dtype=float)
    pos = np.zeros((N,2), dtype=float)
    pos = np.array(pos) #needed to allow subtracting lists for r_ij
    accel = np.zeros((N,2), dtype=float)

    #initialize coords
    algo_choice = 1 # = get_user_input()

    if algo_choice == 1:
        pos = randomize_pos(pos)
    else: 
        pos, vel = plummer_sphere_pos(pos, vel)

    #initialize main plot
    fig, (ax, ax_energy) = plt.subplots(2, 1, figsize=(5, 8), height_ratios=[2,1])
    ax.set_xlim(x_min*2, x_max*2)
    ax.set_ylim(y_min*2, y_max*2)
    ax.set_aspect('equal')
    scat = ax.scatter(pos[:, 0], pos[:, 1], s=2, c='blue')

    #initialize energy plot
    energy_data = []
    ax_energy.set_title('Total Energy')
    ax_energy.set_xlabel('dt')
    ax_energy.set_ylabel('Energy')
    energy_line, = ax_energy.plot([], [], 'r-')

    plt.tight_layout()

    #animation
    animation = FuncAnimation(fig, animate, fargs=(pos, vel, accel,
                                                   scat, start, energy_data, 
                                                   energy_line, ax_energy), 
                              frames=100, interval=50, blit=False)
    plt.show()

    # runtime end
    end = time.time()
    print(f"Runtime:   {end-start[1]:.18f} s")
    return 0

if __name__ == "__main__":
    main()

"""
to do:
0. fix energy leak, perhaps due to acceleration being passed into animate as 0 list always
1. understand total_energy
2. understand plummer model
3. fix math and exploding to infinity (may require another model)
4. time to refactor code and maybe separate into files, getting messy

later:
1. add energy-conserving models: leapfrog, symplectic rkn4, or forest-ruth; compare with overlay under rk4
2. make units real
3. optimize
    -gpu (macos: pytorch with MPS)
    -fast multipole methods (cooked)
    -barnes-hut (goated but maybe cooked)
"""
