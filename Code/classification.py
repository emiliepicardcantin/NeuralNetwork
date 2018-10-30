# -------------------------------------------------------------------------------------------------------------- #
# Mathematics
import numpy as np
import math
from scipy.special import expit

# Visualization 
import matplotlib.pyplot as plt

# Debug and performance
from tqdm import tqdm
from time import time

NEG_INF = -1.79769313e+308

def normalizing_features(X_train, X_test):
    num_examples = X_train.shape[1]
    
    mean_by_feature = np.sum(X_train, axis=1, keepdims=True) / num_examples
    std_deviation = np.sum(X_train**2, axis=1, keepdims=True) / num_examples

    X_train = (X_train - mean_by_feature) / std_deviation
    X_test = (X_test - mean_by_feature) / std_deviation

    return X_train, X_test

# -------------------------------------------------------------------------------------------------------------- #
# Gradient manipulations
def clip_gradients(gradients, max_value=1):
    '''
    Clips the gradients' values between -max_value and max_value.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dW", "db"
    num_layers -- number of hidden layers in the neural network
    max_value -- everything above this number is set to this number, and everything less than -max_value is set to -max_value
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''

    for gradient_name in gradients:
        gradients[gradient_name] = np.clip(gradients[gradient_name], -max_value, max_value)
    
    return gradients
# -------------------------------------------------------------------------------------------------------------- #
# Matrix manipulations
def one_hot_matrix(indices, max_index):
    """
        Argument:
        max_index -- value of the maximum index
        indices -- numpy array of shape (m,) containing values smaller than max_index
        
        Returns:
        ohm -- numpy ndarray of shape (max_index, m) where ohm[i,j] = 1 if indices[j] = i
    """
    one_hot_matrix = np.zeros((max_index, indices.shape[0]))
    for i, val in enumerate(indices):
        one_hot_matrix[val,i] = 1
    return one_hot_matrix
# -------------------------------------------------------------------------------------------------------------- #
# Activation functions
def softmax(z):
    """
        Argument:
        z -- numpy 2D array of shape (n,m)
        
        Returns:
        normalized numpy 2D darray of shape (n, m) 
    """
    t = np.exp(z)
    t_sum = np.sum(t, axis=0)
    return t / t_sum
# -------------------------------------------------------------------------------------------------------------- #
# Parameter initialization
def parameters_init(n_l, mu=0, sigma=0.01):
    """
        Argument:
        n_l -- sizes of the layers (array)
        mu  -- mean to use for the standard normal distribution random sampling
        sigma -- std deviation to use for the standard normal distribution random sampling
        
        Note:
        n_l[0] is the size of the input layer
        n_l[-1] is the size of the output layer
        
        Returns:
        parameters -- python dictionary containing the parameters W and b of each layer in a dictionary:
            Wi -- weight matrix of layer i of shape (n_l[i], n_l[i-1])
            bi -- bias vector of layer i of shape (n_l[i], 1)
    """

    params = {}
    for i in range(1, len(n_l)):
        params["W"+str(i)] = np.random.randn(n_l[i], n_l[i-1]) * sigma
        params["b"+str(i)] = np.zeros((n_l[i], 1))
    
    return params  

def parameters_relu_init(n_l):
    """
        Argument:
        n_l -- sizes of the layers (array)
        
        Note:
        n_l[0] is the size of the input layer
        n_l[-1] is the size of the output layer
        
        Returns:
        parameters -- python dictionary containing the parameters W and b of each layer in a dictionary:
            Wi -- weight matrix of layer i of shape (n_l[i], n_l[i-1])
            bi -- bias vector of layer i of shape (n_l[i], 1)
    """
    params = {}
    for i in range(1, len(n_l)):
        params["W"+str(i)] = np.random.randn(n_l[i], n_l[i-1]) * np.sqrt(2 / n_l[i-1])
        params["b"+str(i)] = np.zeros((n_l[i], 1))
    
    return params  

def parameters_xavier_init(n_l):
    """
        Argument:
        n_l -- sizes of the layers (array)
        
        Note:
        n_l[0] is the size of the input layer
        n_l[-1] is the size of the output layer
        
        Returns:
        parameters -- python dictionary containing the parameters W and b of each layer in a dictionary:
            Wi -- weight matrix of layer i of shape (n_l[i], n_l[i-1])
            bi -- bias vector of layer i of shape (n_l[i], 1)
    """
    params = {}
    for i in range(1, len(n_l)):
        params["W"+str(i)] = np.random.randn(n_l[i], n_l[i-1]) * np.sqrt(1 / n_l[i-1])
        params["b"+str(i)] = np.zeros((n_l[i], 1))
    
    return params  
# -------------------------------------------------------------------------------------------------------------- #
# Forward propagation
def forward_prop_compute_Z(A_prev, W, b):
    return np.dot(W, A_prev) + b

def forward_prop_compute_A(Z, activation_function):
    if activation_function == "tanh":
        A = np.tanh(Z)
    elif activation_function == "sigmoid":
        A = expit(Z)
    elif activation_function == "relu":
        A = np.maximum(Z, 0)
    elif activation_function == "softmax":
        A = softmax(Z)
    else:
        raise ValueError("The activation function "+activation_function+" is not implemented. Choose between tanh and sigmoid.")

    return A

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
        Z = forward_prop_compute_Z(cache["A"+str(layer-1)], params["W"+str(layer)], params["b"+str(layer)])
        # A = forward_prop_compute_A(Z, "tanh")
        A = forward_prop_compute_A(Z, "relu")
        cache["Z"+str(layer)] = Z
        cache["A"+str(layer)] = A
    
    # Output layer is different, using sigmoid function
    Z = forward_prop_compute_Z(cache["A"+str(num_layers-1)], params["W"+str(num_layers)], params["b"+str(num_layers)])
    A_L = forward_prop_compute_A(Z, last_act_fnct)
    cache["Z"+str(num_layers)] = Z
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
    
    m = Y.shape[1] # number of examples
    
    if last_act_fnct == "sigmoid":
        # Compute the cross-entropy cost
        cost = - (np.dot(Y, np.log(A_L).T) / m) - (np.dot(1-Y, np.log(1-A_L).T) / m)
    elif last_act_fnct == "softmax":
        t = np.log(A_L)
        t[t==-np.inf] = NEG_INF
        cost = -1 * np.sum(Y * t) / m
        # cost = - np.dot(np.reshape(Y.T, (1,-1)), np.reshape(np.log(A_L).T,(-1,1))) / m

    # Makes sure cost is the dimension we expect, e.g., turns [[17]] into 17 
    cost = np.asscalar(cost)
    assert(isinstance(cost, float))
    
    return cost

def l2_regularization(num_examples, num_layers, parameters, lambda_param, learning_rate):
    reg = 0
    param_reg = {}

    coeff = (learning_rate*lambda_param/num_examples)
    
    for layer in range(1,num_layers+1):
        reg += np.linalg.norm(parameters["W"+str(layer)]) ** 2
        param_reg["W"+str(layer)] = coeff * parameters["W"+str(layer)]

    cost_reg = (reg * lambda_param) / (2*num_examples)

    return cost_reg, param_reg

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
        
        # dZ = dA_layer * (1 - np.square(cache["A"+str(layer)])) # tanh
        dZ = dA_layer * ((cache["Z"+str(layer)] >= 0) * 1) # relu

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
        params["W"+str(layer)] = params["W"+str(layer)] - learning_rate*VdW_corr / (np.sqrt(SdW_corr) + epsilon)
        params["b"+str(layer)] = params["b"+str(layer)] - learning_rate*Vdb_corr / (np.sqrt(Sdb_corr) + epsilon)
    
    return params, adam_params

def nn_model(
    X, Y, n_l, 
    previous_parameters=None,
    initialization="standard", opt_fnct="standard",
    learning_rate=0.05, lambda_param=0.05, num_iterations=10000, print_cost=None,
    beta1=0.9, beta2=0.999, epsilon=10**(-8),
    model_file=None
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

    num_layers = len(n_l) - 1
    num_examples = X.shape[1]
    num_classes = Y.shape[0]
    
    # Initialize parameters, Inputs: "n_l". Outputs = "W and b parameters by layer".
    if not previous_parameters :
        if initialization == "standard":
            parameters = parameters_init(n_l)
        elif initialization == "xavier":
            parameters = parameters_xavier_init(n_l)
        elif initialization == "relu":
            parameters = parameters_relu_init(n_l)
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
    cost_is_nan = False
    
    for i in range(num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A_L, cache".
        A_L, cache = forward_propagation(X, parameters, num_layers, last_act_fnct=last_act_fnct)

        
        # Cost function. Inputs: "A_L, Y, parameters". Outputs: "cost".
        cost = compute_cost(A_L, Y, parameters, last_act_fnct=last_act_fnct) 
        
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, num_layers, cache, X, Y, last_act_fnct=last_act_fnct)
        grads = clip_gradients(grads)
    
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        if opt_fnct == "adam":
            parameters, adam_params = update_parameters_adam(
                parameters, grads, num_layers, i+1, adam_params,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon
            )
        else:
            parameters = update_parameters(parameters, grads, num_layers, learning_rate=learning_rate)
        
        # Apply regularization 
        cost_reg, param_reg = l2_regularization(num_examples, num_layers, parameters, lambda_param, learning_rate)
        cost = cost + cost_reg
        costs[i] = cost
        for param in param_reg:
            parameters[param] -= param_reg[param]

        # Printing costs
        if print_cost and i % print_cost == 0:
            print("Cost after iteration "+str(i+1)+" : "+str(repr(cost)))
            
        # More prints for debugging
        if math.isnan(cost) or cost == np.nan:
            print("\nCost is NaN for iteration "+str(i+1)+"!")
            print("A_L : min = "+str(np.min(A_L))+", max = "+str(np.max(A_L)))
            print("A_L-1 : min = "+str(np.min(cache["A"+str(num_layers-1)]))+", max = "+str(np.max(cache["A"+str(num_layers-1)])))

            print("Z"+str(num_layers)+" : min = "+str(np.min(cache["Z"+str(num_layers)]))+", max = "+str(np.max(cache["Z"+str(num_layers)])))
            print("W"+str(num_layers)+" : min = "+str(np.min(parameters["W"+str(num_layers)]))+", max = "+str(np.max(parameters["W"+str(num_layers)])))
            print("b"+str(num_layers)+" : min = "+str(np.min(parameters["b"+str(num_layers)]))+", max = "+str(np.max(parameters["b"+str(num_layers)])))
            print("dW"+str(num_layers)+" : min = "+str(np.min(grads["dW"+str(num_layers)]))+", max = "+str(np.max(grads["dW"+str(num_layers)])))
            print("db"+str(num_layers)+" : min = "+str(np.min(grads["db"+str(num_layers)]))+", max = "+str(np.max(grads["db"+str(num_layers)])))
            
            cost_is_nan = True
    
    print("Cost after all iterations : "+str(cost)+"\n")

    if opt_fnct == "adam":
        all_params = {"parameters": parameters, "adam_params":adam_params}
    else:
        all_params = {"parameters": parameters}

    if model_file and not cost_is_nan:
        np.save(model_file, {**all_params, "costs":costs})

    return all_params, costs

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