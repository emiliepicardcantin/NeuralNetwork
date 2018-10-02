import numpy as np
import math
from tqdm import tqdm
from scipy.special import expit
import matplotlib.pyplot as plt

from time import time

def softmax(z):
    t = np.exp(z)
    t_sum = np.sum(t, axis=0)
    return t / t_sum

def initialize_parameters(n_l, seed=42):
    """
        Argument:
        n_l -- sizes of the layers (array)
        
        Note:
        n_l[0] is the size of the input
        n_l[-1] is the size of the output
        
        Returns:
        parameters -- python dictionary containing the parameters W and b of each layer in a dictionary:
            Wi -- weight matrix of layer i of shape (n_l[i], n_l[i-1])
            bi -- bias vector of layer i of shape (n_l[i], 1)
    """

    np.random.seed(seed)
    
    num_layers = len(n_l) - 1
    params = {}
    for i in range(1, num_layers+1):
        params["W"+str(i)] = np.random.randn(n_l[i], n_l[i-1])*0.01
        params["b"+str(i)] = np.zeros((n_l[i], 1))
    
    return params  

def xavier_initialization(n_l, seed=42):
    """
        Argument:
        n_l -- sizes of the layers (array)
        
        Note:
        n_l[0] is the size of the input
        n_l[-1] is the size of the output
        
        Returns:
        parameters -- python dictionary containing the parameters W and b of each layer in a dictionary:
            Wi -- weight matrix of layer i of shape (n_l[i], n_l[i-1])
            bi -- bias vector of layer i of shape (n_l[i], 1)
    """

    np.random.seed(seed)
    
    num_layers = len(n_l) - 1
    params = {}
    for i in range(1, num_layers+1):
        params["W"+str(i)] = np.random.randn(n_l[i], n_l[i-1]) * np.sqrt(1 / n_l[i-1])
        params["b"+str(i)] = np.zeros((n_l[i], 1))
    
    return params  

def forward_prop_compute_z(A_prev, W, b, activation_function):
    z = np.dot(W, A_prev) + b
    if activation_function == "tanh":
        a = np.tanh(z)
    elif activation_function == "sigmoid":
        a = expit(z)
    elif activation_function == "relu":
        a = np.maximum(z, 0)
    elif activation_function == "softmax":
        a = softmax(z)
    else:
        raise ValueError("The activation function "+activation_function+" is not implemented. Choose between tanh and sigmoid.")

    return z,a

def forward_propagation(X, params, num_layers):
    """
        Argument:
        X -- input data of size (n_l[0], m)
        parameters -- python dictionary containing the parameters by layer (output of initialization function)
        
        Returns:
        A_L -- The sigmoid output of the last activation
        cache -- a dictionary containing "Zi" and "Ai" for each layer
    """

    cache = {"A0": X}
    
    # Hidden layers, using tanh activation function
    for layer in range(1,num_layers):   
        z,a = forward_prop_compute_z(cache["A"+str(layer-1)], params["W"+str(layer)], params["b"+str(layer)], "tanh")
        cache["Z"+str(layer)] = z
        cache["A"+str(layer)] = a
    
    # Output layer is different, using sigmoid function
    z,A_L = forward_prop_compute_z(cache["A"+str(num_layers-1)], params["W"+str(num_layers)], params["b"+str(num_layers)], "sigmoid")
    cache["Z"+str(num_layers)] = z
    cache["A"+str(num_layers)] = A_L
    
    return A_L, cache

def compute_cost(A_L, Y, params):
    """
        Computes the cross-entropy cost
        
        Arguments:
        A_L -- The sigmoid output of the last activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        params -- python array containing the parameters W and b by layer
        
        Returns:
        cost -- cross-entropy cost
    """
    
    m = Y.shape[1] # number of example
    
    # Compute the cross-entropy cost
    cost = - (np.dot(Y, np.log(A_L).T) / m) - (np.dot(1-Y, np.log(1-A_L).T) / m)

    # Makes sure cost is the dimension we expect, e.g., turns [[17]] into 17 
    cost = np.asscalar(cost)
    assert(isinstance(cost, float))
    
    return cost

def cost_regularization(num_examples, num_layers, parameters, lambda_param):
    reg = 0
    for layer in range(1,num_layers+1):
        l2_norm = np.square(parameters["W"+str(layer)])
        l2_norm = np.sum(l2_norm, axis=0)
        l2_norm = np.sum(l2_norm)
        reg += l2_norm

    return (reg * lambda_param) / (2*num_examples)

def backward_propagation_derivatives(dA, A, A_prev, m, activation_function, Z=None):
    if activation_function == "sigmoid":
        dZ = dA * A * (1-A)
    elif activation_function == "tanh":
        dZ = dA * (1 - np.square(A))
    elif activation_function == "relu":
        if not Z:
            raise ValueError("The relu activation function requires Z.")
        dZ = dA * (Z >= 0)
    elif activation_function == "softmax":
        dZ = A * (1 - dA)
    else:
        raise ValueError("This activation function is not implemented. Please choose between sigmoid, tanh, or relu.")
    
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    
    return dZ, dW, db

def backward_propagation(params, num_layers, cache, X, Y, last_act_fnct="sigmoid"):
    """
        Implement the backward propagation using the instructions above.
        
        Arguments:
        params -- python dictionary containing the parameters W and b by layer
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (number of examples, number of features)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    grads = {}

    # Initialization of dA_L
    A_L = cache["A"+str(num_layers)]

    if last_act_fnct == "sigmoid":
        dA_L = (A_L - Y) / (A_L * (1-A_L))
    elif last_act_fnct == "softmax":
        dA_L = - Y / A_L

    # Computing dZ_L according to sigmoid actionvation function
    dZ_L, dW_L, db_L = backward_propagation_derivatives(dA_L, A_L, cache["A"+str(num_layers-1)], m, last_act_fnct)
    grads["dZ"+str(num_layers)] = dZ_L
    grads["dW"+str(num_layers)] = dW_L
    grads["db"+str(num_layers)] = db_L
    
    for layer in range(num_layers-1, 0, -1):
        dA_layer = np.dot(params["W"+str(layer+1)].T, grads["dZ"+str(layer+1)])
        dZ, dW, db = backward_propagation_derivatives(dA_layer, cache["A"+str(layer)], cache["A"+str(layer-1)], m, "tanh")
        grads["dZ"+str(layer)] = dZ
        grads["dW"+str(layer)] = dW
        grads["db"+str(layer)] = db
  
    return grads

def update_parameters(params, grads, num_examples, num_layers, lambda_param=0.5, learning_rate = 1.2):
    """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        parameters -- python dictionary containing the parameters W and b by layer
        grads -- python dictionary containing the gradients dZ, dW, db by layer
        
        Returns:
        parameters -- python dictionary containing the updated parameters by layer
    """
    
    # Update rule for each parameter
    for layer in range(1,num_layers+1):
        # params["W"+str(layer)] = (1 - learning_rate*lambda_param/num_examples)*params["W"+str(layer)] - learning_rate*grads["dW"+str(layer)]
        params["W"+str(layer)] = params["W"+str(layer)] - learning_rate*grads["dW"+str(layer)]
        params["b"+str(layer)] = params["b"+str(layer)] - learning_rate*grads["db"+str(layer)]
    
    return params

def nn_model(X, Y, n_l, initialization="standard", learning_rate=0.05, num_iterations=10000, print_cost=False):
    """
        Arguments:
        X -- dataset of shape (number of examples, number of features)
        Y -- labels of shape (1, number of examples)
        n_l -- sizes of the hidden layers (array)
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
    """

    num_examples = X.shape[0]
    num_layers = len(n_l) - 1
    num_classes = Y.shape[0]


    assert(num_examples == Y.shape[1], "Y : "+str(Y.shape))
    
    # Initialize parameters, Inputs: "n_l". Outputs = "W and b parameters by layer".
    if initialization == "standard":
        parameters = initialize_parameters(n_l)
    elif initialization == "xavier":
        parameters = xavier_initialization(n_l)
    else:
        raise ValueError("This type of initialization is not implemented, please choose between standard and tanh.")
    
    # Loop (gradient descent)
    costs = [None for i in range(num_iterations)]
    
    for i in range(num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A_L, cache".
        A_L, cache = forward_propagation(X, parameters, num_layers)
        
        # Cost function. Inputs: "A_L, Y, parameters". Outputs: "cost".
        cost = compute_cost(A_L, Y, parameters) #+ cost_regularization(num_examples,num_layers,parameters,lambda_param=0.5)
        costs[i] = cost
    
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        if num_classes > 1:
            last_act_fnct = "softmax"
        else :
            last_act_fnct = "sigmoid"
        grads = backward_propagation(parameters, num_layers, cache, X, Y, last_act_fnct=last_act_fnct)
    
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, num_examples, num_layers, lambda_param=0.5, learning_rate=learning_rate)
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i+1, cost))
    
    print("Cost after all iterations : "+str(cost)+"\n")

    return parameters, costs

def predict(parameters, X, num_layers):
    """
        Using the learned parameters, predicts a class for each example in X
        
        Arguments:
        parameters -- python array containing the parameters W and b by layer
        X -- input data of size (number of features, number of examples)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A_L, cache = forward_propagation(X, parameters, num_layers)
    # print("A_L : \n"+str(A_L))
    # print("A_L : "+str(A_L.shape))
    # print("Cached As : "+str([layer["A"].shape for layer in cache]))
    # print(np.sum(A_L == A_L[0][0]))
    # print(A_L.shape)
    # print(np.sum(A_L != A_L[0][0]))
    predictions = (A_L > 0.5)
    
    return predictions, A_L

def compute_f1_score(predictions, labels):
    print("Predictions : "+str(predictions.shape))
    all_positives = np.sum(predictions)
    print("All positives : "+str(all_positives))
    true_positives = np.dot(predictions, labels.T)
    print("True positives : "+str(true_positives))

    precision = 1.0
    if all_positives > 0:
        precision = true_positives / all_positives
    print("Precision : "+str(precision))

    false_negatives = np.dot(1-predictions, labels.T)
    print("False negatives : "+str(false_negatives))
    all_negatives = false_negatives + true_positives
    print("All negatives : "+str(all_negatives))

    recall = 1.0
    if all_negatives > 0:
        recall = true_positives / (false_negatives+true_positives)
    print("Recall : "+str(recall))

    return float(2 * precision * recall / (precision + recall))
