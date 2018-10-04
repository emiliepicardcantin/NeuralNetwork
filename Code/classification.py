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

def one_hot_matrix(labels, num_classes):
    num_examples = labels.shape[0]
    one_hot_matrix = np.zeros((num_classes, num_examples))
    for i, label in enumerate(labels):
        one_hot_matrix[label,i] = 1
    return one_hot_matrix

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

def forward_propagation(X, params, num_layers, last_act_fnct="sigmoid"):
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
    z,A_L = forward_prop_compute_z(cache["A"+str(num_layers-1)], params["W"+str(num_layers)], params["b"+str(num_layers)], last_act_fnct)
    cache["Z"+str(num_layers)] = z
    cache["A"+str(num_layers)] = A_L
    
    return A_L, cache

def compute_cost(A_L, Y, params, last_act_fnct="sigmoid"):
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
    
    if last_act_fnct == "sigmoid":
        # Compute the cross-entropy cost
        cost = - (np.dot(Y, np.log(A_L).T) / m) - (np.dot(1-Y, np.log(1-A_L).T) / m)
    elif last_act_fnct == "softmax":
        cost = - np.dot(np.reshape(Y.T, (1,-1)), np.reshape(np.log(A_L).T,(-1,1))) / m

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

def backward_propagation_derivatives(dZ, A_prev, m):
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    
    return dW, db

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

    # Same for both sigmoid and softmax
    dZ_L = A_L - Y

    # Computing dZ_L according to sigmoid actionvation function
    dW_L, db_L = backward_propagation_derivatives(dZ_L, cache["A"+str(num_layers-1)], m)
    grads["dZ"+str(num_layers)] = dZ_L
    grads["dW"+str(num_layers)] = dW_L
    grads["db"+str(num_layers)] = db_L
    
    for layer in range(num_layers-1, 0, -1):
        dA_layer = np.dot(params["W"+str(layer+1)].T, grads["dZ"+str(layer+1)])
        
        dZ = dA_layer * (1 - np.square(cache["A"+str(layer)])) # tanh
        # dZ = dA * (cache["Z"+str(layer)] >= 0) # relu

        dW, db = backward_propagation_derivatives(dZ, cache["A"+str(layer-1)], m)
        grads["dZ"+str(layer)] = dZ
        grads["dW"+str(layer)] = dW
        grads["db"+str(layer)] = db
  
    return grads

def update_parameters(params, grads, num_layers, learning_rate=1.2):
    """
        Updates parameters using the gradient descent update rule
        
        Arguments:
        parameters -- python dictionary containing the parameters W and b by layer
        grads -- python dictionary containing the gradients dZ, dW, db by layer
        
        Returns:
        parameters -- python dictionary containing the updated parameters by layer
    """
    
    # Update rule for each parameter
    for layer in range(1,num_layers+1):
        params["W"+str(layer)] = params["W"+str(layer)] - learning_rate*grads["dW"+str(layer)]
        params["b"+str(layer)] = params["b"+str(layer)] - learning_rate*grads["db"+str(layer)]
    
    return params

def update_parameters_adam(params, grads, num_layers, iteration, adam_params, learning_rate=1.2, beta1=0.9, beta2=0.999, epsilon=10**(-8)):
    """
        Updates parameters using the adam optimization rule 
        
        Arguments:
        parameters -- python dictionary containing the parameters W and b by layer
        grads -- python dictionary containing the gradients dZ, dW, db by layer
        
        Returns:
        parameters -- python dictionary containing the updated parameters by layer
    """
    
    # Update rule for each parameter
    for layer in range(1,num_layers+1):
        # Moment
        adam_params["VdW"+str(layer)] = beta1*adam_params["VdW"+str(layer)] + (1-beta1)*grads["dW"+str(layer)]
        adam_params["Vdb"+str(layer)] = beta1*adam_params["Vdb"+str(layer)] + (1-beta1)*grads["db"+str(layer)]

        # RMS propagation
        adam_params["SdW"+str(layer)] = beta2*adam_params["SdW"+str(layer)] + (1-beta2)*(grads["dW"+str(layer)]**2)
        adam_params["Sdb"+str(layer)] = beta2*adam_params["Sdb"+str(layer)] + (1-beta2)*(grads["db"+str(layer)]**2)

        # Correction
        VdW_corr = adam_params["VdW"+str(layer)] / (1 - beta1**iteration)
        Vdb_corr = adam_params["Vdb"+str(layer)] / (1 - beta1**iteration)
        SdW_corr = adam_params["SdW"+str(layer)] / (1 - beta2**iteration)
        Sdb_corr = adam_params["Sdb"+str(layer)] / (1 - beta2**iteration)

        # Parameters update
        params["W"+str(layer)] = params["W"+str(layer)] - learning_rate*VdW_corr / np.sqrt(SdW_corr) + epsilon
        params["b"+str(layer)] = params["b"+str(layer)] - learning_rate*Vdb_corr / np.sqrt(Sdb_corr) + epsilon
    
    return params, adam_params

def nn_model(
    X, Y, n_l, 
    previous_parameters=None,
    initialization="standard", opt_fnct="standard",
    learning_rate=0.05, num_iterations=10000, print_cost=False,
    beta1=0.9, beta2=0.999, epsilon=10**(-8)
):
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
    
    # Initialize parameters, Inputs: "n_l". Outputs = "W and b parameters by layer".
    if not previous_parameters :
        if initialization == "standard":
            parameters = initialize_parameters(n_l)
        elif initialization == "xavier":
            parameters = xavier_initialization(n_l)
        else:
            raise ValueError("This type of initialization is not implemented, please choose between standard and tanh.")

        if opt_fnct == "adam":
            adam_params = {}
            for layer in range(1, num_layers+1):
                adam_params["VdW"+str(layer)] = 0
                adam_params["Vdb"+str(layer)] = 0
                adam_params["SdW"+str(layer)] = 0
                adam_params["Sdb"+str(layer)] = 0
    else:
        print("Continuing from previous parameters.")
        parameters = previous_parameters["parameters"]

        if opt_fnct == "adam":
            adam_params = previous_parameters["adam_params"]
            
    if num_classes > 1:
        last_act_fnct = "softmax"
    else :
        last_act_fnct = "sigmoid"
        
    # Loop (gradient descent)
    costs = [None for i in range(num_iterations)]
    
    for i in range(num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A_L, cache".
        A_L, cache = forward_propagation(X, parameters, num_layers, last_act_fnct=last_act_fnct)

        # Cost function. Inputs: "A_L, Y, parameters". Outputs: "cost".
        cost = compute_cost(A_L, Y, parameters, last_act_fnct=last_act_fnct) #+ cost_regularization(num_examples,num_layers,parameters,lambda_param=0.5)
        costs[i] = cost
        
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, num_layers, cache, X, Y, last_act_fnct=last_act_fnct)
    
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        if opt_fnct == "adam":
            parameters, adam_params = update_parameters_adam(
                parameters, grads, num_layers, i+1, adam_params,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon
            )
        else:
            parameters = update_parameters(parameters, grads, num_layers, learning_rate=learning_rate)
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i+1, cost))
            # print(parameters)
            # print(cache)
            # print(grads)
            # print("\n\n")
    
    print("Cost after all iterations : "+str(cost)+"\n")

    if opt_fnct == "adam":
        return {"parameters": parameters, "adam_params":adam_params}, costs
    else:
        return {"parameters":parameters}, costs

def predict(parameters, X, num_layers, last_act_fnct="sigmoid"):
    """
        Using the learned parameters, predicts a class for each example in X
        
        Arguments:
        parameters -- python array containing the parameters W and b by layer
        X -- input data of size (number of features, number of examples)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A_L, cache = forward_propagation(X, parameters, num_layers, last_act_fnct=last_act_fnct)

    if last_act_fnct == "sigmoid": 
        predictions = (A_L > 0.5)
    elif last_act_fnct == "softmax":
        predictions = np.argmax(A_L, axis=0)

    return predictions, A_L

def compute_f1_score(predictions, labels):
    results = {
        "positive":{
            "predictions": np.sum(predictions),
            "labels": np.sum(labels),
            "true_predictions": np.sum(np.dot(predictions, labels.T)),
            "false_predictions": np.sum(np.dot(predictions, 1-labels.T))
        },
        "negative": {
            "predictions": np.sum(1 - predictions),
            "labels": np.sum(1 - labels),
            "true_predictions": np.sum(np.dot(1-predictions, 1-labels.T)),
            "false_predictions": np.sum(np.dot(1-predictions, labels.T))
        }
    }
    f1_score = {
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 0.0
    }

    if results["positive"]["predictions"] > 0:
        f1_score["precision"] = results["positive"]["true_predictions"] / results["positive"]["predictions"]

    if results["positive"]["labels"] > 0:
        f1_score["recall"] = results["positive"]["true_predictions"] / results["positive"]["labels"]

    if f1_score["precision"] > 0.0 and f1_score["recall"] > 0.0:
        f1_score["f1_score"] = float(2 * f1_score["precision"] * f1_score["recall"] / (f1_score["precision"] + f1_score["recall"]))

    return results,f1_score

def compute_f1_score_multi_class(predictions, labels, num_classes):
    results = [None for c in range(num_classes)]
    f1_scores = [None for c in range(num_classes)]

    for c in range(num_classes):
        r,f1 = compute_f1_score((predictions == c)*1, (labels == c)*1)
        results[c] = r
        f1_scores[c] = f1
        # print("Class "+str(c)+" : "+str(f1))


    return results, f1_scores