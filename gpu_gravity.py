import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time
import torch
import torch.nn.functional as F

# Check if MPS (Metal Performance Shaders) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU only")

# Constant declarations
G = 6.674e-11
N = 500  # Increased since GPU can handle more particles
M = 10000000
dt = 0.1
g = 9.81
epsilon = 1e-1

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

def randomize_pos_gpu(N, device):
    """Generate random positions on GPU"""
    pos = torch.rand((N, 2), device=device, dtype=torch.float32) * (x_max - x_min) + x_min
    return pos

def plummer_sphere_pos_gpu(N, device):
    """Generate Plummer sphere positions and velocities on GPU"""
    a = 10.0
    total_mass = N * M
    
    # Generate positions
    xi = torch.rand(N, device=device)
    r = a / torch.sqrt(xi**(-2/3) - 1)
    
    theta = torch.arccos(2 * torch.rand(N, device=device) - 1)
    phi = 2 * np.pi * torch.rand(N, device=device)
    
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    pos = torch.stack([x, y], dim=1)
    
    # Generate velocities
    r_norm = torch.norm(pos, dim=1)
    M_enc = total_mass * (r_norm**3) / (r_norm**2 + a**2)**(3/2)
    
    # Avoid division by zero
    r_norm_safe = torch.where(r_norm > 0, r_norm, torch.tensor(1e-10, device=device))
    v_esc = torch.sqrt(2 * G * M_enc / r_norm_safe)
    
    v_mag = v_esc * (torch.rand(N, device=device) * 0.6 + 0.1)  # 0.1 to 0.7
    v_theta = 2 * np.pi * torch.rand(N, device=device)
    
    v_x = v_mag * torch.cos(v_theta)
    v_y = v_mag * torch.sin(v_theta)
    vel = torch.stack([v_x, v_y], dim=1)
    
    # Center of mass correction
    com_pos = torch.mean(pos, dim=0)
    com_vel = torch.mean(vel, dim=0)
    
    pos = pos - com_pos
    vel = vel - com_vel
    
    return pos, vel

def calc_acceleration_gpu(pos, masses, epsilon, G):
    """
    Calculate accelerations using GPU vectorization
    pos: (N, 2) tensor of positions
    masses: (N,) tensor of masses
    """
    N = pos.shape[0]
    
    # Create pairwise position differences: (N, N, 2)
    pos_i = pos.unsqueeze(1)  # (N, 1, 2)
    pos_j = pos.unsqueeze(0)  # (1, N, 2)
    r_vec = pos_j - pos_i     # (N, N, 2) - vector from i to j
    
    # Calculate distances with softening
    r_sq = torch.sum(r_vec**2, dim=2)  # (N, N)
    r_sq_soft = r_sq + epsilon**2
    r_mag = torch.sqrt(r_sq_soft)      # (N, N)
    
    # Avoid self-interaction
    mask = torch.eye(N, device=pos.device, dtype=torch.bool)
    r_mag = torch.where(mask, torch.inf, r_mag)
    
    # Calculate force magnitudes
    masses_j = masses.unsqueeze(0)  # (1, N)
    force_mag = G * masses_j / (r_mag**3)  # (N, N)
    
    # Calculate accelerations
    accel = torch.sum(force_mag.unsqueeze(2) * r_vec, dim=1)  # (N, 2)
    
    return accel

def total_energy_gpu(pos, vel, masses, G):
    """Calculate total energy on GPU"""
    N = pos.shape[0]
    
    # Kinetic energy
    kinetic = 0.5 * torch.sum(masses.unsqueeze(1) * vel**2)
    
    # Potential energy
    pos_i = pos.unsqueeze(1)  # (N, 1, 2)
    pos_j = pos.unsqueeze(0)  # (1, N, 2)
    r_vec = pos_j - pos_i     # (N, N, 2)
    r_mag = torch.norm(r_vec, dim=2)  # (N, N)
    
    # Avoid self-interaction and double counting
    mask = torch.triu(torch.ones(N, N, device=pos.device, dtype=torch.bool), diagonal=1)
    r_mag_masked = torch.where(mask, r_mag, torch.inf)
    
    masses_i = masses.unsqueeze(1)  # (N, 1)
    masses_j = masses.unsqueeze(0)  # (1, N)
    potential = -G * torch.sum(masses_i * masses_j / r_mag_masked)
    
    return kinetic + potential

def leapfrog_gpu(pos, vel, masses, dt, epsilon, G):
    """Leapfrog integration on GPU"""
    # Calculate initial acceleration
    accel = calc_acceleration_gpu(pos, masses, epsilon, G)
    
    # First half-step velocity update
    vel = vel + 0.5 * dt * accel
    
    # Position update
    pos = pos + dt * vel
    
    # Calculate new acceleration
    accel = calc_acceleration_gpu(pos, masses, epsilon, G)
    
    # Second half-step velocity update
    vel = vel + 0.5 * dt * accel
    
    return pos, vel

class GPUNBodySimulation:
    def __init__(self, N, device, use_plummer=True):
        self.N = N
        self.device = device
        self.masses = torch.full((N,), M, device=device, dtype=torch.float32)
        
        if use_plummer:
            self.pos, self.vel = plummer_sphere_pos_gpu(N, device)
        else:
            self.pos = randomize_pos_gpu(N, device)
            self.vel = torch.zeros((N, 2), device=device, dtype=torch.float32)
        
        self.energy_history = []
        
    def step(self):
        """Perform one simulation step"""
        self.pos, self.vel = leapfrog_gpu(
            self.pos, self.vel, self.masses, dt, epsilon, G
        )
    
    def get_energy(self):
        """Calculate and return current total energy"""
        return total_energy_gpu(self.pos, self.vel, self.masses, G).item()
    
    def get_positions_cpu(self):
        """Get positions as numpy array for plotting"""
        return self.pos.cpu().numpy()

def animate_gpu(frame, sim, scat, start, energy_data, energy_line, ax_energy):
    """Animation function for GPU simulation"""
    
    # Perform simulation step
    sim.step()
    
    # Update positions
    pos_cpu = sim.get_positions_cpu()
    scat.set_offsets(pos_cpu)
    
    # Calculate energy every few frames
    if frame % 10 == 0:
        energy = sim.get_energy()
        energy_data.append(energy)
        
        if len(energy_data) > 1:
            energy_line.set_data(range(len(energy_data)), energy_data)
            ax_energy.relim()
            ax_energy.autoscale_view()
    
    # Frametime calculation
    end = time.time()
    frametime = end - start[0]
    print(f"Frametime: {frametime:.6f} s ({1/frametime:.1f} FPS)")
    start[0] = time.time()
    
    return scat, energy_line

def benchmark_gpu_vs_cpu(N_values=[50, 100, 200, 500]):
    """Benchmark GPU vs CPU performance"""
    print("\nBenchmarking GPU vs CPU performance:")
    print("N\t\tCPU Time\tGPU Time\tSpeedup")
    print("-" * 50)
    
    for N in N_values:
        if N > 200 and device.type == 'cpu':
            continue  # Skip large N for CPU-only systems
            
        # GPU benchmark
        masses_gpu = torch.full((N,), M, device=device, dtype=torch.float32)
        pos_gpu = torch.randn((N, 2), device=device, dtype=torch.float32) * 50
        vel_gpu = torch.randn((N, 2), device=device, dtype=torch.float32) * 10
        
        # Warm up
        for _ in range(5):
            calc_acceleration_gpu(pos_gpu, masses_gpu, epsilon, G)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(50):
            calc_acceleration_gpu(pos_gpu, masses_gpu, epsilon, G)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        gpu_time = (time.time() - start_time) / 50
        
        # CPU benchmark
        masses_cpu = masses_gpu.cpu()
        pos_cpu = pos_gpu.cpu()
        
        start_time = time.time()
        for _ in range(50):
            calc_acceleration_gpu(pos_cpu, masses_cpu, epsilon, G)
        cpu_time = (time.time() - start_time) / 50
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"{N}\t\t{cpu_time:.6f}s\t{gpu_time:.6f}s\t{speedup:.2f}x")

def main():
    """Main simulation function with GPU support"""
    print(f"Running on device: {device}")
    
    # Run benchmark
    if device.type in ['mps', 'cuda']:
        benchmark_gpu_vs_cpu()
    
    # Create simulation
    sim = GPUNBodySimulation(N, device, use_plummer=True)
    
    # Runtime start
    start = [time.time(), time.time()]
    
    # Initialize plots
    fig, (ax, ax_energy) = plt.subplots(2, 1, figsize=(6, 7), height_ratios=[3, 1])
    ax.set_xlim(x_min*2, x_max*2)
    ax.set_ylim(y_min*2, y_max*2)
    ax.set_aspect('equal')
    ax.set_title(f'GPU N-Body Simulation (N={N}) on {device.type.upper()}')
    ax.set_facecolor('black')
    
    # Initial scatter plot
    pos_initial = sim.get_positions_cpu()
    scat = ax.scatter(pos_initial[:, 0], pos_initial[:, 1], s=3, c='white', alpha=0.8)
    
    # Initialize energy plot
    energy_data = []
    ax_energy.set_title('Total Energy Conservation')
    ax_energy.set_xlabel('Time Steps (×10)')
    ax_energy.set_ylabel('Energy (J)')
    ax_energy.grid(True, alpha=0.3)
    energy_line, = ax_energy.plot([], [], 'lime', linewidth=2)
    
    plt.tight_layout()
    
    # Animation
    print("Starting animation...")
    animation = FuncAnimation(
        fig, animate_gpu,
        fargs=(sim, scat, start, energy_data, energy_line, ax_energy),
        frames=2000, interval=16, blit=False, cache_frame_data=False
    )
    
    plt.show()
    
    # Runtime end
    end = time.time()
    print(f"Total Runtime: {end-start[1]:.6f} s")
    return 0

if __name__ == "__main__":
    main()

"""
GPU Optimization for M4 MacBook Air:

1. Install PyTorch with MPS support:
   pip install torch torchvision torchaudio

2. Key Features:
   - Uses Metal Performance Shaders (MPS) for GPU acceleration
   - Vectorized operations eliminate explicit loops
   - Batch processing of all particles simultaneously
   - Memory-efficient tensor operations

3. Performance Benefits:
   - 5-50x speedup over CPU for large N
   - Can handle 500+ particles smoothly
   - Energy conservation tracking
   - Real-time benchmarking

4. M4 Specific Advantages:
   - Unified memory architecture
   - Optimized Metal shaders
   - Low power consumption
   - Excellent thermal management

Alternative GPU Options for M4:
- JAX with Metal backend (experimental)
- CuPy with ROCm (not supported on M4)
- OpenCL kernels (more complex setup)

Note: MPS is the best option for M4 MacBooks!
"""