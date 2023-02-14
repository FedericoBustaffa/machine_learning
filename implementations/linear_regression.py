import matplotlib.pyplot as plt
import numpy as np


def loss(points, a, b):
    loss = 0.0
    n = len(points)

    for i in range(n):
        x = points[0][i]
        y = points[1][i]
        loss += (y - (a * x + b)) ** 2

    return loss / float(len(points))


def fit(points, a, b, L):
    n = len(points)
    
    a_gradient = 0
    b_gradient = 0
    
    for i in range(n):
        x = points[0][i]
        y = points[1][i]

        a_gradient += -(2 / n) * x * (y - (a * x + b))
        b_gradient += -(2 / n) * (y - (a * x + b))

    new_a = a - a_gradient * L
    new_b = b - b_gradient * L

    return new_a, new_b


# main
a = 0.0
b = 0.0
L = 0.00001

points = np.array([
    [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
    [ 1, 2.4, 2.7, 3.4, 5.5, 6.6, 6.8, 8.1, 8, 11 ]
])

for _ in range(100000):
    a, b = fit(points, a, b, L)

print(loss(points, a, b))
print(a, b)

x = np.linspace(points.min() - 1, points.max() + 1, 100)
y = a * x + b

# plot
plt.title("Regressione Lineare")

plt.scatter(points[0], points[1], color="black")
plt.plot(x, y, color="red")

plt.show()

