import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

def f(x):
    return np.sin(3*x) + 0.5*x

def df(x):
    return 3*np.cos(3*x) + 0.5



#the formula is x(k + 1) = x(k) - (f(x(k)) / (df(x(k)) / dx))
def newton_single(x0, tol=1e-6, max_iter=7):
    points = [x0]
    for _ in range(max_iter):
        xk = points[-1]
        yk = f(xk)
        dyk = df(xk)
        if abs(dyk) < 1e-12:
            break
        x_next = xk - yk/dyk
        points.append(x_next)
        if abs(f(x_next)) < tol:
            break
    return points

x0 = 1.0
points = newton_single(x0)

x = np.linspace(-3, 3, 400)
y = f(x)

plt.figure(figsize=(12, 7))
plt.plot(x, y, label='f(x) = sin(3x) + 0.5x', color='blue')

colors = ['red', 'orange', 'green', 'purple', 'brown', 'cyan', 'magenta']

for i in range(len(points) - 1):
    xk = points[i]
    yk = f(xk)
    slope = df(xk)
    
    x_tangent = np.linspace(xk - 0.6, xk + 0.6, 50)
    y_tangent = slope * (x_tangent - xk) + yk
    plt.plot(x_tangent, y_tangent, color=colors[i % len(colors)], linestyle='--', label=f'Tangent at x_{i}')
    
    plt.plot(xk, yk, 'o', color=colors[i % len(colors)], markersize=8)
    plt.text(xk, yk + 0.3, f'{i}', color=colors[i % len(colors)], fontsize=12, fontweight='bold')
    
    plt.plot(points[i+1], 0, 'x', color=colors[i % len(colors)], markersize=10)
    plt.text(points[i+1], -0.5, f'{i+1}', color=colors[i % len(colors)], fontsize=12, fontweight='bold')

plt.plot(points[-1], f(points[-1]), 'o', color='black', markersize=8, label='Final approx')
plt.text(points[-1], f(points[-1]) + 0.3, f'{len(points)-1}', color='black', fontsize=12, fontweight='bold')

plt.axhline(0, color='black', linewidth=0.7)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Newton's Method Visualization on Wave-like Function with Iteration Numbers")
plt.legend()
plt.grid(True)
plt.show()
