"""
Disclaimer: The datareader helper function has been reproduced from work by Yann LeCun.
"""

import sys
import struct
import numpy as np
import math


## read images and labels from binary files, one by one (generator function) ##
def datareader(imagename,labelname):
    with open(imagename,"rb") as images, open(labelname,"rb") as labels:
        magic,length,w,h=struct.unpack('>IIII',images.read(4*4))
        magic,lengthL=struct.unpack('>II',labels.read(4*2))
        print("dataset of {} images of size {}x{}".format(length,w,h),file=sys.stderr)
        assert length == lengthL
        for i in range(length):
            yield np.frombuffer(images.read(w*h),dtype='uint8'),labels.read(1)[0]

def softmax(weights,vector,bias):
    # activation function between network layers
    return np.tanh(np.matmul(weights,vector)+bias)

def step_fwd(v_input,w_matrices,bias_vectors):
    # propagation of a single input through the network
    num_layers = len(w_matrices)
    activation_values = []
    vector_values = [v_input]
    for i in range(num_layers):
        v_input = softmax(w_matrices[i],v_input,bias_vectors[i])
        activation_values.append(v_input)
        vector_values.append(v_input)

    return v_input,activation_values,vector_values

def backpropagate(output,control,w_matrices,activation_values):
    # propagation of a single prediction error back through the network
    # we take the final step explicitely
    diff = output-control
    E = 0.5*np.sum(np.square(diff)) # error function
    grad = np.multiply(-diff,np.ones(np.shape(activation_values[-1]))-np.square(activation_values[-1]))
    gradients = []
    gradients.append(grad)

    # stepping backwards
    for i in range(len(w_matrices)-1,0,-1):
        grad = np.multiply(np.matmul(grad,w_matrices[i]),np.ones(np.shape(activation_values[i-1]))-np.square(activation_values[i-1]))       
        gradients.append(grad)

    return E,list(reversed(gradients))

def update_weights(w_matrices,gradients,vector_values,learning_rate):
    # for now, no updates are made to the biases

    for i in range(len(w_matrices)):
        w_matrices[i] = w_matrices[i]+learning_rate*np.outer(gradients[i],vector_values[i])
    return w_matrices

#num_training_samples = len(list(datareader("train-images-idx3-ubyte","train-labels-idx1-ubyte")))
#num_test_samples = len(list(datareader("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")))

layer_sizes = [784,784,28,10]   # input and output layers are included, so the number of hidden layers is len-2
learning_rate = 0.01

weight_matrices = []
bias_vectors = []
# Generates initial random weights
for i in range(0,len(layer_sizes)-1):
    weights = np.random.normal(0.0,2.0/layer_sizes[i+1],size=(layer_sizes[i+1],layer_sizes[i]))
    bias = np.full(layer_sizes[i+1],0.0)
    weight_matrices.append(weights)
    bias_vectors.append(bias)

# The first implementation is going to be a fully stochastic gradient descent
# Training
k = 0
for image,label in datareader("train-images-idx3-ubyte","train-labels-idx1-ubyte"):

    rescaled = np.divide(image,255)
    control = np.full(10,-1.0)
    control[label] = 1.0

    out_vec,act_val,vec_val = step_fwd(rescaled,weight_matrices,bias_vectors)
    E,grad_val = backpropagate(out_vec,control,weight_matrices,act_val)
    weight_matrices = update_weights(weight_matrices,grad_val,vec_val,learning_rate)
    
    k += 1
    if k > 10000:
        print(out_vec-control)
        break

# Testing
i_am_confusion = np.zeros((10,10))
class_numbers = np.zeros(10,dtype=int)

l = 0
for testimage,testlabel in datareader("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte"):
    rescaled = np.divide(testimage,255)
    out_vec,_,_ = step_fwd(rescaled,weight_matrices,bias_vectors)
    class_numbers[testlabel] += 1   # increment the correct class
    i_am_confusion[testlabel,np.argmax(out_vec)] += 1
    l += 1
    if l > 1000:
        break

for i in range(10):
    i_am_confusion[i,:] = np.divide(i_am_confusion[i,:],class_numbers[i])

# evaluation of results and implementation of bias and/or different hidden layer sizes is to follow.