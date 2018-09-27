import numpy as np
import math
from tqdm import tqdm
from scipy.special import expit

# Defines the random seed used, 42 is always the answer.
np.random.seed(42)

def initialize_parameters(n_l):
    """
    Argument:
    n_l -- sizes of the layers (array)
    
    Note:
    n_l[0] is the size of the input
    n_l[-1] is the size of the output
    
    Returns:
    parameters -- python array containing the parameters W and b of each layer in a dictionary:
        Wi -- weight matrix of layer i of shape (n_l[i], n_l[i-1])
        bi -- bias vector of layer i of shape (n_l[i], 1)
    """
    
    num_layers = len(n_l) - 1
    params = [
        {
            "W": np.random.randn(n_l[i+1], n_l[i])*0.01,
            "b": np.zeros((n_l[i+1], 1))
        }
        for i in range(num_layers)
    ]
    
    return params  

def forward_prop_compute_z(A_prev, W, b, activation_function):
    z = np.dot(W, A_prev) + b
    if activation_function == "tanh":
        a = np.tanh(z)
    elif activation_function == "sigmoid":
        a = expit(z)
    else:
        raise ValueError("The activation function "+activation_function+" is not implemented. Choose between tanh and sigmoid.")

    return z,a

def forward_propagation(X, params):
    """
    Argument:
    X -- input data of size (n_l[0], m)
    parameters -- python array containing the parameters by layer (output of initialization function)
    
    Returns:
    A_L -- The sigmoid output of the last activation
    cache -- an array containing "Zi" and "Ai" for each layer
    """

    cache = []
    
    # First hidden layer
    z,a = forward_prop_compute_z(X, params[0]["W"], params[0]["b"], "tanh")
    cache.append({"Z":z, "A":a})
    
    # Subsequent hidden layer
    for layer in range(1,len(params)-1):   
        z,a = forward_prop_compute_z(cache[-1]["A"], params[layer]["W"], params[layer]["b"], "tanh")
        cache.append({"Z": z, "A":a})
    
    # Output layer is different, using sigmoid function
    z,A_L = forward_prop_compute_z(cache[-1]["A"], params[len(params)-1]["W"], params[len(params)-1]["b"], "sigmoid")
    cache.append({"Z":z, "A":A_L})
    
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
    # print("A_L : "+str(A_L.shape))
    # print("Y : "+str(Y.shape))

    # Compute the cross-entropy cost
    # print("First part of cost : "+str(- (np.dot(Y, np.log(A_L).T) / m)))
    # print("Second part of cost : "+str(- (np.dot(1-Y, np.log(1-A_L).T) / m)))
    cost = - (np.dot(Y, np.log(A_L).T) / m) - (np.dot(1-Y, np.log(1-A_L).T) / m)

    cost = np.asscalar(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 

    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(params, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    params -- python array containing the parameters W and b by layer
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (number of examples, number of features)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    grads = [None for layer in params]
    
    num_layers = len(params)
    assert(num_layers == len(cache))
    
    dZ_L = cache[-1]["A"] - Y
    dW_L = np.dot(dZ_L, cache[-2]["A"].T) / m
    db_L = np.sum(dZ_L, axis=1, keepdims=True) / m
    grads[num_layers-1] = {"dZ":dZ_L, "dW":dW_L, "db":db_L}
    
    for layer in range(num_layers-1, 1, -1):
        dZ = np.dot(params[layer]["W"].T, grads[layer]["dZ"]) * (1 - np.power(cache[layer-1]["A"], 2))
        dW = np.dot(dZ, cache[layer-2]["A"].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        grads[layer-1] = {"dZ":dZ, "dW":dW, "db":db}
        
    dZ = np.dot(params[1]["W"].T, grads[1]["dZ"]) * (1 - np.power(cache[0]["A"], 2))
    dW = np.dot(dZ, X.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    grads[0] = {"dZ":dZ, "dW":dW, "db":db}
  
    return grads

def update_parameters(params, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python array containing the parameters W and b by layer
    grads -- python array containing the gradients dZ, dW, db by layer
    
    Returns:
    parameters -- python array containing the updated parameters by layer
    """
    
    # Update rule for each parameter
    num_layers = len(params)
    assert(num_layers == len(grads))
    
    for layer in range(num_layers):
        params[layer]["W"] -= learning_rate*grads[layer]["dW"]
        params[layer]["b"] -= learning_rate*grads[layer]["db"]
    
    return params

def nn_model(X, Y, n_l, learning_rate=0.05, num_iterations=10000, print_cost=False):
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
    
    # Initialize parameters, Inputs: "n_l". Outputs = "W and b parameters by layer".
    parameters = initialize_parameters(n_l)
    
    # Loop (gradient descent)
    tqdm_range = tqdm(range(num_iterations))
    for i in tqdm_range:
        # Forward propagation. Inputs: "X, parameters". Outputs: "A_L, cache".
        # print("Forward propagation")
        A_L, cache = forward_propagation(X, parameters)
        # print("A_L : \n"+str(A_L))
        
        # Cost function. Inputs: "A_L, Y, parameters". Outputs: "cost".
        # print("Computing cost")
        cost = compute_cost(A_L, Y, parameters)
        # print("\tCost = "+str(cost))

        tqdm_range.set_description("Iteration "+str(i)+", cost = "+str(cost))

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        # print("Backward propagation")
        grads = backward_propagation(parameters, cache, X, Y)
        # print("\tGrads = "+str(grads))
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        # print("Updating parameters")
        parameters = update_parameters(parameters, grads, learning_rate)
        # print("\tParameters : "+str(parameters))
        
        # # Print the cost every 1000 iterations
        # if print_cost and i % 100 == 0:
        #     print("Cost after iteration %i: %f" %(i, cost)+"\n")

    # print("Cost after all iterations : "+str(cost)+"\n")
    return parameters

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python array containing the parameters W and b by layer
    X -- input data of size (number of features, number of examples)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A_L, cache = forward_propagation(X, parameters)
    # print("A_L : \n"+str(A_L))
    # print("A_L : "+str(A_L.shape))
    # print("Cached As : "+str([layer["A"].shape for layer in cache]))
    # print(np.sum(A_L == A_L[0][0]))
    # print(A_L.shape)
    # print(np.sum(A_L != A_L[0][0]))
    predictions = (A_L > 0.5)
    
    return predictions

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


def predict_multiclass(parameters_by_class, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python array containing the parameters W and b by layer by class
    X -- input data of size (number of features, number of examples)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    num_classes = len(parameters_by_class)
    num_examples = X.shape[1]

    A_L_by_class = np.zeros((num_classes, num_examples))
    for i in range(num_classes):
        A_L, cache = forward_propagation(X, parameters_by_class[i])
        A_L_by_class[i,:] = A_L

    predictions = np.argmax(A_L_by_class, axis=0)
    
    return predictions