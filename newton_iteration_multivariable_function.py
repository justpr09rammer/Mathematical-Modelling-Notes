import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

def F(vec):
    x, y = vec
    return np.array([
        x**2 + y**2 - 4,         
        np.exp(x) + y - 1         
    ])

def jacobian(vec):
    x, y = vec
    return np.array([
        [2*x,        2*y],
        [np.exp(x),     1]
    ])

def newton_system(x0, tol=1e-6, max_iter=10):
    xk = np.array(x0, dtype=float)
    points = [xk.copy()]
    for i in range(max_iter):
        J = jacobian(xk)
        Fx = F(xk)
        try:
            delta = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            print("Jacobian is singular!")
            break
        
        xk = xk - delta
        points.append(xk.copy())
        if np.linalg.norm(F(xk)) < tol:
            break
    return np.array(points)

x0 = [1.5, 1.5]
points = newton_system(x0, max_iter=7)

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
x_grid, y_grid = np.meshgrid(x, y)

z1 = x_grid**2 + y_grid**2 - 4
z2 = np.exp(x_grid) + y_grid - 1

fig = plt.figure(figsize=(14, 7))

# Plot f1 surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x_grid, y_grid, z1, cmap='viridis', alpha=0.7)
ax1.set_title('Surface of f1(x, y) = x² + y² - 4')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f1(x, y)')
ax1.grid(False)

ax1.plot(points[:, 0], points[:, 1], F(points.T)[0], 'ro-', label='Newton steps')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_grid, y_grid, z2, cmap='plasma', alpha=0.7)
ax2.set_title('Surface of f2(x, y) = e^x + y - 1')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f2(x, y)')
ax2.grid(False)

ax2.plot(points[:, 0], points[:, 1], F(points.T)[1], 'ro-', label='Newton steps')
ax2.legend()

plt.suptitle('Newton Method Root Finding for System of Equations', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
