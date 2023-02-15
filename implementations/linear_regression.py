import matplotlib.pyplot as plt
import numpy as np


def loss(points, a, b):
    loss = 0.0
    n = len(points[0])

    for i in range(n):
        x = points[0][i]
        y = points[1][i]
        loss += (y - (a * x + b)) ** 2

    return loss / float(n)


def fit(points, a, b, L):
    n = len(points[0])

    a_delta = 0.0
    b_delta = 0.0

    for i in range(n):
        x = points[0][i]
        y = points[1][i]

        a_delta += (2 / n) * (y - (a * x + b)) * (-x)
        b_delta += (2 / n) * (y - (a * x + b)) * (-1)

    new_a = a - a_delta * L
    new_b = b - b_delta * L

    return new_a, new_b


# main
a = 0.0
b = 0.0
L = 0.0000001

points = np.array([
    [i for i in range(10)],
    [np.random.randint(0, 10) for i in range(10)]
])

# for i in range(100000):
# a, b = fit(points, a, b, L)

while True:
    l1 = loss(points, a, b)
    a, b = fit(points, a, b, L)
    l2 = loss(points, a, b)
    if l1 % 111:
        print(l1)
    if l1 < l2:
        break

x = np.linspace(points.min() - 1, points.max() + 1, 30)
y = a * x + b

plt.title("Regressione Lineare")
plt.scatter(points[0], points[1], color="black")
plt.ylim(-10, 30)
plt.plot(x, y, color="red")

plt.show()


# print(f"Iterations {i}")
print(f"a: {a}\nb: {b}")
print(f"Loss: {loss(points, a, b)}")


# plot
