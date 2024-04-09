import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Make sure PyTorch uses GPU if available
device = torch.device("mps")

# Constants
G = 2.67430e-11  # Gravitational constant

# Initial conditions with tensors on the specified device (GPU if available)
m = torch.tensor([1.0e24, 1.0e24, 1.0e24], dtype=torch.float, device=device)
r = torch.tensor([[0, 0], [150e9, 0], [300e9, 0]], dtype=torch.float, device=device)
v = torch.tensor([[0, 15e3], [0, -10e3], [0, 5e3]], dtype=torch.float, device=device)

dt = torch.tensor(10000.0, dtype=torch.float, device=device)  # Time step

# This function updates positions and velocities
def update_positions_and_velocities(r, v, m):
    acceleration = torch.zeros_like(r)
    for i in range(3):
        for j in range(i + 1, 3):
            r_diff = r[j] - r[i]
            dist = torch.norm(r_diff)
            force_direction = r_diff / dist
            # Newton's law of universal gravitation
            force = G * m[i] * m[j] / (dist ** 2)
            # Update accelerations based on force
            acceleration[i] += force * force_direction / m[i]
            acceleration[j] -= force * force_direction / m[j]
    v += acceleration * dt
    r += v * dt
    return r, v

# Prepare plotting
fig, ax = plt.subplots()
points, = ax.plot([], [], 'o')
ax.set_xlim(-5e11, 5e11)
ax.set_ylim(-5e11, 5e11)

def init():
    points.set_data([], [])
    return points,

def animate(i):
    global r, v
    r, v = update_positions_and_velocities(r, v, m)
    points.set_data(r[:, 0].cpu().numpy(), r[:, 1].cpu().numpy())
    return points,

ani = FuncAnimation(fig, animate, frames=200, init_func=init, blit=True, interval=20)
plt.show()
