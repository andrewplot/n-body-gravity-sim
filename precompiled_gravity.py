import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time
from numba import jit, njit, prange

# Constant declarations
G = 6.674e-11
N = 500
M = 10000000
dt = 0.1
g = 9.81
epsilon = 1e-1  # helps with not approaching infinity
mass = np.full(N, M, dtype=np.float64)

x_min = y_min = -100
x_max = y_max = 100

def get_user_input():
    while True:
        user_input = input("1 for random, 2 for plummer sphere: ")
        try:
            user_input = int(user_input)
            if user_input in [1, 2]:
                return user_input
            else:
                print("Err: Invalid input")
        except ValueError:
            print("Err: NaN input")

def randomize_pos(pos):
    for i in range(N):
        pos[i] = [random.uniform(x_min, x_max), random.uniform(y_min, y_max)]
    return pos

def plummer_sphere_pos(pos, vel):
    a = 10.0  # scale radius
    total_mass = N * M
    
    # Generate positions
    for i in range(N):
        xi = random.random()
        r = a / np.sqrt((xi**(-2/3) - 1))
        
        theta = np.arccos(2 * random.random() - 1)
        phi = 2 * np.pi * random.random()
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        
        pos[i] = [x, y]
    
    # Generate velocities
    for i in range(N):
        r_i = np.linalg.norm(pos[i])
        M_enc = total_mass * (r_i**3) / (r_i**2 + a**2)**(3/2)
        v_esc = np.sqrt(2 * G * M_enc / r_i) if r_i > 0 else 0
        v_mag = v_esc * random.uniform(0.1, 0.7)
        
        v_theta = 2 * np.pi * random.random()
        v_x = v_mag * np.cos(v_theta)
        v_y = v_mag * np.sin(v_theta)
        
        vel[i] = [v_x, v_y]
    
    # Center of mass correction
    com_pos = np.mean(pos, axis=0)
    com_vel = np.mean(vel, axis=0)
    
    for i in range(N):
        pos[i] = pos[i] - com_pos
        vel[i] = vel[i] - com_vel
    
    return pos, vel

@njit
def total_energy_numba(mass, vel, pos):
    """Numba-compiled energy calculation"""
    # Kinetic energy
    kinetic = 0.0
    for i in range(N):
        kinetic += 0.5 * mass[i] * (vel[i, 0]**2 + vel[i, 1]**2)
    
    # Potential energy
    potential = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            r_ij = np.sqrt(dx*dx + dy*dy)
            potential -= G * mass[i] * mass[j] / r_ij
    
    return kinetic + potential

@njit
def calc_accel_numba(i, pos, mass):
    """Numba-compiled acceleration calculation for particle i"""
    ax = 0.0
    ay = 0.0
    
    for j in range(N):
        if i == j:
            continue
            
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        r_mag_sq = dx*dx + dy*dy
        
        if r_mag_sq == 0:
            continue
            
        # With softening parameter
        r_mag_sq_soft = r_mag_sq + epsilon*epsilon
        r_mag = np.sqrt(r_mag_sq_soft)
        r_mag3 = r_mag_sq_soft * r_mag
        
        force_factor = G * mass[j] / r_mag3
        ax += force_factor * dx
        ay += force_factor * dy
    
    return ax, ay

@njit
def calc_all_accel_numba(pos, mass, accel):
    """Calculate accelerations for all particles"""
    for i in range(N):
        accel[i, 0], accel[i, 1] = calc_accel_numba(i, pos, mass)

@njit
def leapfrog_numba(pos, vel, accel, mass):
    """Numba-compiled leapfrog integration"""
    # First half-step velocity update
    for i in range(N):
        vel[i, 0] += 0.5 * dt * accel[i, 0]
        vel[i, 1] += 0.5 * dt * accel[i, 1]
    
    # Position update
    for i in range(N):
        pos[i, 0] += dt * vel[i, 0]
        pos[i, 1] += dt * vel[i, 1]
    
    # Calculate new accelerations
    calc_all_accel_numba(pos, mass, accel)
    
    # Second half-step velocity update
    for i in range(N):
        vel[i, 0] += 0.5 * dt * accel[i, 0]
        vel[i, 1] += 0.5 * dt * accel[i, 1]

@njit(parallel=True)
def calc_all_accel_parallel(pos, mass, accel):
    """Parallel version of acceleration calculation"""
    for i in prange(N):
        ax = 0.0
        ay = 0.0
        
        for j in range(N):
            if i == j:
                continue
                
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            r_mag_sq = dx*dx + dy*dy
            
            if r_mag_sq == 0:
                continue
                
            r_mag_sq_soft = r_mag_sq + epsilon*epsilon
            r_mag = np.sqrt(r_mag_sq_soft)
            r_mag3 = r_mag_sq_soft * r_mag
            
            force_factor = G * mass[j] / r_mag3
            ax += force_factor * dx
            ay += force_factor * dy
        
        accel[i, 0] = ax
        accel[i, 1] = ay

@njit
def leapfrog_parallel(pos, vel, accel, mass):
    """Leapfrog with parallel acceleration calculation"""
    # First half-step velocity update
    for i in range(N):
        vel[i, 0] += 0.5 * dt * accel[i, 0]
        vel[i, 1] += 0.5 * dt * accel[i, 1]
    
    # Position update
    for i in range(N):
        pos[i, 0] += dt * vel[i, 0]
        pos[i, 1] += dt * vel[i, 1]
    
    # Calculate new accelerations (parallel)
    calc_all_accel_parallel(pos, mass, accel)
    
    # Second half-step velocity update
    for i in range(N):
        vel[i, 0] += 0.5 * dt * accel[i, 0]
        vel[i, 1] += 0.5 * dt * accel[i, 1]

def animate(frame, pos, vel, accel, scat, start, energy_data, energy_line, ax_energy, use_parallel=True):
    """Animation function with Numba optimization"""
    
    if use_parallel:
        leapfrog_parallel(pos, vel, accel, mass)
    else:
        leapfrog_numba(pos, vel, accel, mass)
    
    # Energy calculation (every few frames to save computation)
    if frame % 5 == 0:  # Calculate energy every 5 frames
        energy = total_energy_numba(mass, vel, pos)
        energy_data.append(energy)
        
        energy_line.set_data(range(len(energy_data)), energy_data)
        ax_energy.relim()
        ax_energy.autoscale_view()
    
    # Update scatter plot
    scat.set_offsets(pos)
    
    # Frametime calculation
    end = time.time()
    print(f"Frametime: {end-start[0]:.6f} s")
    start[0] = time.time()
    
    return scat, energy_line

def main():
    """Main simulation function"""
    # Runtime start
    start = [time.time(), time.time()]
    
    # Array declarations (ensure contiguous arrays for Numba)
    pos = np.zeros((N, 2), dtype=np.float64, order='C')
    vel = np.zeros((N, 2), dtype=np.float64, order='C')
    accel = np.zeros((N, 2), dtype=np.float64, order='C')
    
    # Initialize coordinates
    algo_choice = 1  # get_user_input()
    
    if algo_choice == 2:
        pos = randomize_pos(pos)
    else:
        pos, vel = plummer_sphere_pos(pos, vel)
    
    # Convert to numpy arrays with proper dtype
    pos = np.array(pos, dtype=np.float64)
    vel = np.array(vel, dtype=np.float64)
    
    # Calculate initial accelerations
    calc_all_accel_numba(pos, mass, accel)
    
    # Compile functions by running them once (warm-up)
    print("Warming up Numba functions...")
    pos_temp = pos.copy()
    vel_temp = vel.copy()
    accel_temp = accel.copy()
    
    leapfrog_numba(pos_temp, vel_temp, accel_temp, mass)
    total_energy_numba(mass, vel_temp, pos_temp)
    print("Numba compilation complete!")
    
    # Initialize plots
    fig, (ax, ax_energy) = plt.subplots(2, 1, figsize=(5, 8), height_ratios=[2, 1])
    ax.set_xlim(x_min*2, x_max*2)
    ax.set_ylim(y_min*2, y_max*2)
    ax.set_aspect('equal')
    ax.set_title(f'N-Body Simulation (N={N}) with Numba')
    scat = ax.scatter(pos[:, 0], pos[:, 1], s=1, c='blue')
    
    # Initialize energy plot
    energy_data = []
    ax_energy.set_title('Total Energy')
    ax_energy.set_xlabel('Time Steps (×5)')
    ax_energy.set_ylabel('Energy')
    energy_line, = ax_energy.plot([], [], 'r-')
    
    plt.tight_layout()
    
    # Animation
    animation = FuncAnimation(
        fig, animate,
        fargs=(pos, vel, accel, scat, start, energy_data, energy_line, ax_energy, True),
        frames=1000, interval=1, blit=False, cache_frame_data=False
    )
    plt.show()
    
    # Runtime end
    end = time.time()
    print(f"Total Runtime: {end-start[1]:.6f} s")
    return 0

if __name__ == "__main__":
    main()

"""
Numba Optimization Notes:
1. Install with: pip install numba
2. @njit decorator compiles functions to machine code
3. @njit(parallel=True) with prange() enables parallel execution
4. Use contiguous numpy arrays (order='C') for best performance
5. Avoid Python objects inside jitted functions
6. First run includes compilation time - subsequent runs are much faster

Performance improvements:
- 10-100x speedup typical for numerical code
- Parallel version can utilize multiple CPU cores
- Memory layout optimization for cache efficiency
"""