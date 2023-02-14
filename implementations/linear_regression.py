import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


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
L = 0.00000001

points = np.array([
    [i for i in range(10)],
    [np.random.randint(0, 10) for i in range(10)]
])

for i in range(1000000):
    a, b = fit(points, a, b, L)

# i = 0
# while True:
#     i += 1
#     l1 = loss(points, a, b)
#     a, b = fit(points, a, b, L)
#     l2 = loss(points, a, b)
#     # print(l1)
#     if (l1 < l2):
#         break

# print(f"Iterations {i}")
print(f"a: {a}\nb: {b}")
print(f"Loss: {loss(points, a, b)}")

model = LinearRegression()
model.fit(points[0].reshape(-1, 1), points[1].reshape(-1, 1))

x = np.linspace(points.min() - 1, points.max() + 1, 30)
y = a * x + b

# plot
plt.title("Regressione Lineare")
plt.ylim(-10, 40)

plt.scatter(points[0], points[1], color="black")
plt.plot(x, y, color="red")
plt.plot(x, model.predict(x.reshape(-1, 1)), color="green")
plt.show()
