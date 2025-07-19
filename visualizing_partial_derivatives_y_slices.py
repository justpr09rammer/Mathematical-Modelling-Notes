import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('dark_background')

# Function and derivative wrt y
def f(x, y):
    return 3 - x**2 - y**2 - x

def fy(x, y):
    return -2*y

# Grid
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
x_grid, y_grid = np.meshgrid(x, y)
z_grid = f(x_grid, y_grid)

# Points to slice at different x values
x_values = [-1.5, -0.75, 0, 0.75, 1.5]
colors = ['cyan', 'lime', 'yellow', 'orange', 'magenta']

# Plot setup
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#111111')
fig.patch.set_facecolor('#111111')

# Main surface
ax.plot_surface(x_grid, y_grid, z_grid, cmap='coolwarm', alpha=0.4, edgecolor='none')

# Draw slices and tangents along y for each fixed x
for x_val, col in zip(x_values, colors):
    y_slice = np.linspace(-2, 2, 100)
    z_slice = f(x_val, y_slice)
    tangent_slope = fy(x_val, 0)
    z0 = f(x_val, 0)
    tangent = z0 + tangent_slope * (y_slice - 0)
    
    ax.plot(x_val * np.ones_like(y_slice), y_slice, z_slice, color=col, linewidth=2, label=f'x = {x_val}')
    ax.plot(x_val * np.ones_like(y_slice), y_slice, tangent, color=col, linestyle='--', linewidth=1)

# Labels and view
ax.set_xlabel('x', labelpad=15)
ax.set_ylabel('y', labelpad=15)
ax.set_zlabel('z', labelpad=10)
ax.set_title('Multiple y-slices at different x values', fontsize=14, color='white')
ax.view_init(elev=28, azim=-58)
ax.legend(loc='upper right', fontsize=9)

ax.grid(False)
ax.tick_params(colors='white')

plt.tight_layout()
plt.show()
