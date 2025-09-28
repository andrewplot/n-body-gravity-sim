"""
nbody_portfolio.py

A polished N-body demo with:
 - vectorized acceleration
 - RK4 and Leapfrog integrators
 - Matplotlib interactive UI: integrator selector, IC selector, pause/resume, reset, speed slider
 - Energy plotting for the active integrator
 - Radial density plot for Plummer initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button, Slider
import random
import time

# -------------------------
# Constants & Parameters
# -------------------------
G = 6.674e-11  # gravitational constant (kept symbolic; values may be scaled)
N = 50         # number of particles (reduce to 30-50 for interactivity)
M = 1e7        # mass per particle (uniform for simplicity)
dt_base = 0.05  # base dt; scale by speed slider in UI
softening = 0.5  # softening length to avoid singularities (tweakable)
mass = np.full(N, M)

# coordinate bounds for visualization
x_min = y_min = -100.0
x_max = y_max = 100.0

# -------------------------
# Helper: vectorized physics
# -------------------------
def pairwise_displacements(positions):
    """
    Return displacement array r[i,j] = pos[j] - pos[i] shape (N,N,2)
    """
    return positions[None, :, :] - positions[:, None, :]

def calc_accelerations(positions, masses, G=G, soft=softening):
    """
    Vectorized acceleration calculation:
      a_i = G * sum_j m_j * (r_j - r_i) / (|r_j - r_i|^2 + soft^2)^(3/2)
    positions: (N,2)
    returns: accelerations (N,2)
    """
    r = pairwise_displacements(positions)  # r[i,j] = pos[j] - pos[i]
    dist2 = np.sum(r * r, axis=2) + soft**2  # (N,N)
    inv_dist3 = dist2 ** (-1.5)
    # zero self-interaction
    np.fill_diagonal(inv_dist3, 0.0)
    # multiply by mass_j for each j
    m_j = masses[None, :, None]  # (1,N,1)
    accel = G * np.sum(m_j * r * inv_dist3[:, :, None], axis=1)  # (N,2)
    return accel

def total_energy(positions, velocities, masses, G=G, soft=softening):
    """
    Compute total energy (kinetic + potential) vectorized.
    Potential uses pairwise sums i<j.
    """
    # kinetic
    v2 = np.sum(velocities**2, axis=1)
    kinetic = 0.5 * np.sum(masses * v2)

    # potential
    r = pairwise_displacements(positions)
    dist = np.sqrt(np.sum(r*r, axis=2) + soft**2)
    # take upper triangle i<j
    i_indices, j_indices = np.triu_indices(positions.shape[0], k=1)
    potential = -np.sum(G * masses[i_indices] * masses[j_indices] / dist[i_indices, j_indices])

    return kinetic + potential

# -------------------------
# Initial conditions
# -------------------------
def randomize_pos(N, bounds):
    x_min, x_max, y_min, y_max = bounds
    pos = np.column_stack((
        np.random.uniform(x_min, x_max, size=N),
        np.random.uniform(y_min, y_max, size=N)
    ))
    vel = np.zeros((N, 2), dtype=float)
    return pos, vel

def plummer_sphere(N, M_particle, a=10.0):
    """
    Simple Plummer sphere sampling for positions and approximate bound velocities.
    Returns pos (N,2) and vel (N,2).
    """
    pos = np.zeros((N, 2))
    vel = np.zeros((N, 2))
    total_mass = N * M_particle

    for i in range(N):
        # sample radius r from Plummer cumulative distribution
        xi = random.random()
        r = a / np.sqrt(xi**(-2./3.) - 1.)
        # isotropic angles in 3D, then project to 2D plane (z ignored)
        theta = np.arccos(2 * random.random() - 1)
        phi = 2 * np.pi * random.random()
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        pos[i] = [x, y]

    # approximate velocities: fraction of local escape velocity
    for i in range(N):
        r_i = np.linalg.norm(pos[i])
        # enclosed mass approx using Plummer cumulative mass
        M_enc = total_mass * (r_i**3) / (r_i**2 + a**2)**1.5 if r_i > 0 else total_mass / 2.0
        v_esc = np.sqrt(2 * G * M_enc / (r_i + 1e-6)) if r_i > 0 else 0.0
        v_mag = v_esc * random.uniform(0.05, 0.6)
        theta_v = 2 * np.pi * random.random()
        vel[i] = [v_mag * np.cos(theta_v), v_mag * np.sin(theta_v)]

    # remove net COM drift
    pos -= np.mean(pos, axis=0)
    vel -= np.mean(vel, axis=0)
    return pos, vel

def two_galaxy_collision(N, M_particle):
    """
    Create two Plummer spheres separated and moving toward each other.
    """
    half = N // 2
    pos1, vel1 = plummer_sphere(half, M_particle, a=8.0)
    pos2, vel2 = plummer_sphere(N-half, M_particle, a=8.0)
    # translate second galaxy
    offset = np.array([60.0, 0.0])
    pos2 += offset
    # give bulk velocities so they collide
    vel1 += np.array([10.0, 0.0])   # to the right
    vel2 += np.array([-10.0, 0.0])  # to the left
    pos = np.vstack((pos1, pos2))
    vel = np.vstack((vel1, vel2))
    # center-of-mass correction
    pos -= np.mean(pos, axis=0)
    vel -= np.mean(vel, axis=0)
    return pos, vel

# -------------------------
# Integrators
# -------------------------
def leapfrog_step(pos, vel, masses, dt):
    # kick-drift-kick
    accel = calc_accelerations(pos, masses)
    vel_half = vel + 0.5 * dt * accel
    pos_new = pos + dt * vel_half
    accel_new = calc_accelerations(pos_new, masses)
    vel_new = vel_half + 0.5 * dt * accel_new
    return pos_new, vel_new

def rk4_step(pos, vel, masses, dt):
    # k1
    k1_x = vel
    k1_v = calc_accelerations(pos, masses)

    # k2
    x2 = pos + 0.5 * dt * k1_x
    v2 = vel + 0.5 * dt * k1_v
    k2_x = v2
    k2_v = calc_accelerations(x2, masses)

    # k3
    x3 = pos + 0.5 * dt * k2_x
    v3 = vel + 0.5 * dt * k2_v
    k3_x = v3
    k3_v = calc_accelerations(x3, masses)

    # k4
    x4 = pos + dt * k3_x
    v4 = vel + dt * k3_v
    k4_x = v4
    k4_v = calc_accelerations(x4, masses)

    pos_new = pos + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    vel_new = vel + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return pos_new, vel_new

# Map integrator name to function
INTEGRATORS = {
    "RK4": rk4_step,
    "Leapfrog": leapfrog_step
}

# -------------------------
# Simulation State (for reset)
# -------------------------
class SimState:
    def __init__(self, positions, velocities, masses):
        self.pos0 = positions.copy()
        self.vel0 = velocities.copy()
        self.mass = masses.copy()
        self.reset()

    def reset(self):
        self.pos = self.pos0.copy()
        self.vel = self.vel0.copy()
        self.time = 0.0

# -------------------------
# Visualization + UI
# -------------------------
def make_simulation(initial_choice='random', integrator_choice='Leapfrog'):
    # initialize positions and velocities
    bounds = (x_min, x_max, y_min, y_max)
    if initial_choice == 'random':
        pos0, vel0 = randomize_pos(N, bounds)
    elif initial_choice == 'plummer':
        pos0, vel0 = plummer_sphere(N, M)
    elif initial_choice == 'collision':
        pos0, vel0 = two_galaxy_collision(N, M)
    else:
        pos0, vel0 = randomize_pos(N, bounds)

    state = SimState(pos0, vel0, mass)
    return state, integrator_choice

def run_gui():
    # initial setup
    ic_choice = 'plummer'   # default
    integrator_choice = 'Leapfrog'
    state, current_integrator = make_simulation(ic_choice, integrator_choice)

    # prepare figure
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1])
    ax_sim = fig.add_subplot(gs[0, :])
    ax_energy = fig.add_subplot(gs[1, :])
    ax_radial = fig.add_subplot(gs[2, 0])
    ax_controls = fig.add_subplot(gs[2, 1:])
    ax_controls.axis('off')

    scat = ax_sim.scatter(state.pos[:, 0], state.pos[:, 1], s=8)
    ax_sim.set_xlim(x_min * 1.5, x_max * 1.5)
    ax_sim.set_ylim(y_min * 1.5, y_max * 1.5)
    ax_sim.set_aspect('equal')
    ax_sim.set_title("N-body simulation")

    energy_time = []
    energy_data = {name: [] for name in INTEGRATORS.keys()}  # keep per-integrator history

    energy_line, = ax_energy.plot([], [], label='Energy')
    ax_energy.set_xlabel('Steps')
    ax_energy.set_ylabel('Total Energy')
    ax_energy.legend()

    # radial density (for plummer)
    radial_line, = ax_radial.plot([], [], '-o', markersize=3)
    ax_radial.set_title("Radial density (Plummer)")
    ax_radial.set_xlabel("radius")
    ax_radial.set_ylabel("counts")

    # UI widgets: use radio buttons and sliders in ax_controls using subplot positions
    # We'll create small axes within ax_controls for each widget
    left = 0.63
    bottom = 0.08
    width = 0.32
    height = 0.25
    ax_integrator = fig.add_axes([left, bottom + 0.18, width, 0.12])
    radio_integrator = RadioButtons(ax_integrator, tuple(INTEGRATORS.keys()), active=1)

    ax_ic = fig.add_axes([left, bottom + 0.02, width, 0.12])
    radio_ic = RadioButtons(ax_ic, ('random', 'plummer', 'collision'), active=1)

    ax_pause = fig.add_axes([0.02, 0.02, 0.08, 0.05])
    btn_pause = Button(ax_pause, 'Pause', hovercolor='0.975')

    ax_reset = fig.add_axes([0.12, 0.02, 0.08, 0.05])
    btn_reset = Button(ax_reset, 'Reset', hovercolor='0.975')

    ax_speed = fig.add_axes([0.25, 0.02, 0.25, 0.03])
    slider_speed = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0)

    # stateful flags
    anim_running = [True]  # use list to modify from inner funcs

    def update_energy_and_plots():
        # record energy for the current integrator
        E = total_energy(state.pos, state.vel, state.mass)
        energy_data[current_integrator].append(E)
        energy_time.append(len(energy_time))
        energy_line.set_data(range(len(energy_data[current_integrator])), energy_data[current_integrator])
        ax_energy.relim(); ax_energy.autoscale_view()

        # radial density if plummer
        if radio_ic.value_selected == 'plummer':
            r = np.linalg.norm(state.pos, axis=1)
            bins = np.linspace(0, 50, 25)
            hist, edges = np.histogram(r, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            radial_line.set_data(centers, hist)
            ax_radial.relim(); ax_radial.autoscale_view()
        else:
            radial_line.set_data([], [])

    # event callbacks
    def integrator_changed(label):
        nonlocal current_integrator
        current_integrator = label
        print(f"Integrator switched to: {label}")

    def ic_changed(label):
        nonlocal state, energy_time
        print(f"Initial condition switched to: {label} -- resetting")
        state, _ = make_simulation(label, current_integrator)
        # clear energy history for fresh run
        for k in energy_data.keys():
            energy_data[k] = []
        energy_time = []
        scat.set_offsets(state.pos)
        ax_sim.relim()
        ax_sim.autoscale_view()

    def pause_clicked(event):
        if anim_running[0]:
            anim.event_source.stop()
            btn_pause.label.set_text("Resume")
            anim_running[0] = False
        else:
            anim.event_source.start()
            btn_pause.label.set_text("Pause")
            anim_running[0] = True

    def reset_clicked(event):
        print("Reset clicked")
        state.reset()
        for k in energy_data.keys():
            energy_data[k] = []
        energy_time.clear()
        scat.set_offsets(state.pos)

    radio_integrator.on_clicked(integrator_changed)
    radio_ic.on_clicked(ic_changed)
    btn_pause.on_clicked(pause_clicked)
    btn_reset.on_clicked(reset_clicked)

    # animation function
    frame_count = [0]
    start_time = [time.time()]

    def animate(frame):
        frame_count[0] += 1
        dt = dt_base * slider_speed.val

        # one simulation step using chosen integrator
        step_fn = INTEGRATORS[current_integrator]
        state.pos, state.vel = step_fn(state.pos, state.vel, state.mass, dt)
        state.time += dt

        # update scatter
        scat.set_offsets(state.pos)

        # update diagnostics
        update_energy_and_plots()

        # print a frametime occasionally
        if frame_count[0] % 20 == 0:
            now = time.time()
            print(f"Step {frame_count[0]}, elapsed {now - start_time[0]:.3f}s")
            start_time[0] = now

        return scat, energy_line, radial_line

    anim = FuncAnimation(fig, animate, interval=30, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_gui()
