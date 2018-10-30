import unittest
import numpy as np

from time import time

import classification as CL

class TestMatrixManipulations(unittest.TestCase):
    def test_one_hot_matrix(self):
        one_hot_matrix = CL.one_hot_matrix(np.array([0,1,2,3,1]), 4)
        expected_matrix = np.array([[1,0,0,0,0], [0,1,0,0,1], [0,0,1,0,0], [0,0,0,1,0]])
        np.testing.assert_array_equal(one_hot_matrix, expected_matrix)

class TestActivationFunctions(unittest.TestCase):
    def test_softmax(self):
        z = np.array([[1,2,3.1], [4,-5,0]])
        a = np.round(CL.softmax(z), 5)
        expected_a = np.round(np.array([
            [np.exp(1)/(np.exp(1)+np.exp(4)), np.exp(2)/(np.exp(2)+np.exp(-5)),  np.exp(3.1)/(np.exp(3.1)+1)],
            [np.exp(4)/(np.exp(1)+np.exp(4)), np.exp(-5)/(np.exp(2)+np.exp(-5)), 1/(np.exp(3.1)+1)]
            ]),5)
        np.testing.assert_array_equal(a, expected_a)
        np.testing.assert_array_equal(np.sum(a, axis=0, keepdims=True), np.ones((1,3)))

    def test_softmax_with_nan(self):
        z = np.array([np.full(3,-np.infty), np.full(3,-np.infty)])
        a = np.round(CL.softmax(z), 5)
        print(a)

class TestParameterInitialization(unittest.TestCase):
    def test_parameters_init(self):
        params = CL.parameters_init([2, 3, 3, 1])
        param_shapes = {p: params[p].shape for p in params}
        expected_parameter_shapes = {
            'W1': (3,2), 'b1': (3,1),
            'W2': (3,3), 'b2': (3,1),
            'W3': (1,3), 'b3': (1,1)
        }
        self.assertDictEqual(param_shapes, expected_parameter_shapes)
    
    def test_parameters_xavier_init(self):
        params = CL.parameters_xavier_init([2, 3, 3, 1])
        param_shapes = {p: params[p].shape for p in params}
        expected_parameter_shapes = {
            'W1': (3,2), 'b1': (3,1),
            'W2': (3,3), 'b2': (3,1),
            'W3': (1,3), 'b3': (1,1)
        }
        self.assertDictEqual(param_shapes, expected_parameter_shapes)

class TestForwardPropagation(unittest.TestCase):
    def test_forward_prop_compute_Z(self):
        W = np.array([[1,13,20,43], [76, 0, 4, 23], [9, 57, 12, 10]])
        A_prev = np.array([[2,13], [0.1, 0.45], [3, 9], [0.01, 0.5]])
        b = np.array([[1], [2], [3]])

        expected_Z = np.array([[64.73, 221.35], [166.23, 1037.5], [62.8, 258.65]])

        Z = CL.forward_prop_compute_Z(A_prev, W, b)
        np.testing.assert_array_almost_equal(Z, expected_Z)

    def test_forward_prop_compute_A(self):
        Z = np.array([[-1, 0.004, 0.76], [7, -15, -0.9]])

        A = np.round(CL.forward_prop_compute_A(Z, 'tanh'), 4)
        expected_A = np.array([[-0.7616, 0.004, 0.6411], [1.0, -1.0, -0.7163]])
        np.testing.assert_array_equal(A, expected_A)

        A = np.round(CL.forward_prop_compute_A(Z, 'sigmoid'), 4)
        expected_A = np.array([[0.2689, 0.5010, 0.6814], [0.9991, 0.0, 0.2891]])
        np.testing.assert_array_equal(A, expected_A)

        A = np.round(CL.forward_prop_compute_A(Z, 'relu'), 4)
        expected_A = np.array([[0, 0.004, 0.76], [7, 0, 0]])
        np.testing.assert_array_equal(A, expected_A)

        A = np.round(CL.forward_prop_compute_A(Z, 'softmax'), 4)
        expected_A = np.array([[0.0003, 1.0, 0.8402], [0.9997, 0.0, 0.1598]])
        np.testing.assert_array_equal(A, expected_A)

class TestClassification(unittest.TestCase):

    def test_initialize_parameters_sqrt_tanh(self):
        n_l = [2, 2, 2, 1]
        params = CL.xavier_initialization(n_l)
        expected_parameters = {
            'W1': np.array([[ 0.35122995, -0.09776762],[ 0.45798496,  1.07694474]]), 
            'b1': np.array([[0.],[0.]]),
            'W2': np.array([[-0.16557144, -0.16555983],[ 1.11667209,  0.5426583 ]]), 
            'b2': np.array([[0.],[0.]]),
            'W3': np.array([[-0.33196852,  0.38364789]]), 
            'b3': np.array([[0.]])
        }
        for i in range(1,4):
            np.testing.assert_allclose(np.round(params["W"+str(i)],6), np.round(expected_parameters["W"+str(i)],6))
            np.testing.assert_allclose(np.round(params["b"+str(i)],6), np.round(expected_parameters["b"+str(i)],6))

    def test_cost_regularization(self):
        parameters = [
            {
                'W': np.array([[ 0.35122995, -0.09776762],[ 0.45798496,  1.07694474]]), 
                'b': np.array([[0.],[0.]])
            },
            {
                'W': np.array([[-0.16557144, -0.16555983],[ 1.11667209,  0.5426583 ]]), 
                'b': np.array([[0.],[0.]])
            },
            {
                'W': np.array([[-0.33196852,  0.38364789]]), 
                'b': np.array([[0.]])
            }
        ]
        regularization = CL.cost_regularization(10, 3, parameters, 0.5)
        print(regularization)

    
        
    

if __name__ == '__main__':
    unittest.main()