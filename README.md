# Linear Neuron

This neuron it's a modification of the classic perceptron. The  perceptron 
only do binaries classifications, meanwhile this neuron do linear regression. 
This is because the preceptron has an activation function that makes discret 
outputs, the linear neuron delets this function to make a continuos output.

### This neuron has 3 main methods to train itself:
* Stocastic Gradient Descend **(SGD)**
* Batch Gradient Descend **(BGD)**
* Direct

The **SGD** and **BGD** algorithms are iterative, for every iteration, **SGD** goes trough each pattern of the input and calculates every single prediction, meanwhile **BGD** calculates all input predictions in a single iteration. Thats why **SGD** has more random behavior. The **direct** method is not iterative. 

## Libraries
* numpy
* matplotlib
* pandas
