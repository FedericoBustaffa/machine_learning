import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x_values = []
y_values = []

model = LinearRegression()

# plotting
for _ in range(500):
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    x_values.append(np.random.randint(-9, 9))
    y_values.append(np.random.randint(-9, 9))

    x = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values).reshape(-1, 1)
    
    model.fit(x, y)

    plt.scatter(x, y, color="black")
    plt.plot(x, model.predict(x), color="red")
    
    plt.pause(0.00001)

plt.show()
