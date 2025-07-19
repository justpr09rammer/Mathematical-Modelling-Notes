import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('dark_background')

def f(x, y):
    return 3 - x**2 - y**2 - x

def fx(x, y):
    return -2*x - 1

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
x_grid, y_grid = np.meshgrid(x, y)
z_grid = f(x_grid, y_grid)

y_values = [-1.5, -0.75, 0, 0.75, 1.5]
colors = ['cyan', 'lime', 'yellow', 'orange', 'magenta']

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#111111')
fig.patch.set_facecolor('#111111')

ax.plot_surface(x_grid, y_grid, z_grid, cmap='coolwarm', alpha=0.4, edgecolor='none')

for y_val, col in zip(y_values, colors):
    x_slice = np.linspace(-2, 2, 100)
    z_slice = f(x_slice, y_val)
    tangent_slope = fx(0, y_val)
    z0 = f(0, y_val)
    tangent = z0 + tangent_slope * (x_slice - 0)
    
    ax.plot(x_slice, y_val * np.ones_like(x_slice), z_slice, color=col, linewidth=2, label=f'y = {y_val}')
    ax.plot(x_slice, y_val * np.ones_like(x_slice), tangent, color=col, linestyle='--', linewidth=1)

ax.set_xlabel('x', labelpad=15)
ax.set_ylabel('y', labelpad=15)
ax.set_zlabel('z', labelpad=10)
ax.set_title('Multiple x-slices at different y values', fontsize=14, color='white')
ax.view_init(elev=28, azim=-58)
ax.legend(loc='upper right', fontsize=9)

ax.grid(False)
ax.tick_params(colors='white')

plt.tight_layout()
plt.show()
