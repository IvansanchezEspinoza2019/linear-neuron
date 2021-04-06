import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_neuron import LinearNeuron

"""
##### THIS IS A SHORTER DEMOSTRATION EXAMPLE
p = 100

x = -1 + 2 * np.random.rand(p).reshape(1, -1)
y = -18 * x + 6 +15 *np.random.rand(p)
                
print(x)

neuron = LinearNeuron(1, 0.1)

neuron.fit(x, y,solver='a', epochs=200)

xn = np.array([[-1, 1]])
plt.plot(xn.ravel(), neuron.predict(xn), '--k')

"""
    
#### read dataset #####
dataset = pd.read_csv("DataSet1.csv")

## creating the X,Y for training ###
x_train = np.array([dataset.iloc[:, 0]])  # vector size: 1 x 99
y_train = np.array([dataset.iloc[:, 1]])  # vector size: 1 x 99

### linear neuron ##
linear_regression = LinearNeuron(1, 0.1) # 1 dimension problem, 0.1 neuron learning factor

### plot a dataset ###
plt.title("LINEAR REGRESSION")
print(x_train.shape)
print(y_train.shape)

# plot points
plt.scatter(x_train, y_train, label='Train Dataset') 

### training ###
# you can choose between 'SGD', 'BGD' and 'DIRECT' algorithms
# receive:   X: NxP;    Y: 1xP
linear_regression.fit(x_train, y_train, solver='SGD', epochs=400) 

### making the line regression ###
x_vector = np.array([[0, 1]])  # create points between 0 and 1
y_vector = linear_regression.predict(x_vector)  # predict the x vector

print(x_vector.shape)
print(y_vector.shape)

### plot lineal regression ###
plt.plot(x_vector.ravel(), y_vector, '--k', label="Regression", linewidth=2)

### making predictions ###
x_pred = np.array([[0.5, 0.2, 0.4, 0.87, 0.6, 0.75, 0.15]]) # new data for prediction
y_pred = linear_regression.predict(x_pred)                  # predict new data
        
## plot the prediction ###
plt.scatter(x_pred.ravel(), y_pred, label="New Data")

plt.xlabel("X AXIS")
plt.ylabel("Y AXIS")

plt.grid()
plt.legend()

