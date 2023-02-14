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
    [ 1, -1, 1.5, 1, -2, -1, 1, 0, -2.5, 1.45 ]
])

x = np.linspace(0, 11, 100)

for i in range(1000):
    a, b = fit(points, a, b, L)
    # print("%.9f,\t%.9f" % (a, b))
    y = a * x + b
    print(y[0])
    if i % 5 == 0:
        plt.clf()
        plt.ylim(-10, 10)

        plt.scatter(points[0], points[1], color="black")
        plt.plot(x, y, color="red")
        plt.pause(0.000001)

plt.show()

