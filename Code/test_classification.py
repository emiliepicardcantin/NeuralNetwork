import unittest
import numpy

import classification as CL

class TestClassification(unittest.TestCase):
    def test_initialize_parameters(self):
        n_l = [2, 2, 2, 1]
        params = CL.initialize_parameters(n_l)
        expected_parameters = {
            'W1': numpy.array([[ 0.00496714, -0.00138264],[ 0.00647689,  0.0152303 ]]),
            'b1': numpy.array([[0.],[0.]]),
            'W2': numpy.array([[-0.00234153, -0.00234137],[ 0.01579213,  0.00767435]]), 
            'b2': numpy.array([[0.],[0.]]),
            'W3': numpy.array([[-0.00469474,  0.0054256 ]]), 
            'b3': numpy.array([[0.]])
        }
        for i in range(1,4):
            numpy.testing.assert_allclose(numpy.round(params["W"+str(i)],6), numpy.round(expected_parameters["W"+str(i)],6))
            numpy.testing.assert_allclose(numpy.round(params["b"+str(i)],6), numpy.round(expected_parameters["b"+str(i)],6))
            
    def test_initialize_parameters_sqrt_tanh(self):
        n_l = [2, 2, 2, 1]
        params = CL.xavier_initialization(n_l)
        expected_parameters = {
            'W1': numpy.array([[ 0.35122995, -0.09776762],[ 0.45798496,  1.07694474]]), 
            'b1': numpy.array([[0.],[0.]]),
            'W2': numpy.array([[-0.16557144, -0.16555983],[ 1.11667209,  0.5426583 ]]), 
            'b2': numpy.array([[0.],[0.]]),
            'W3': numpy.array([[-0.33196852,  0.38364789]]), 
            'b3': numpy.array([[0.]])
        }
        for i in range(1,4):
            numpy.testing.assert_allclose(numpy.round(params["W"+str(i)],6), numpy.round(expected_parameters["W"+str(i)],6))
            numpy.testing.assert_allclose(numpy.round(params["b"+str(i)],6), numpy.round(expected_parameters["b"+str(i)],6))

    def test_cost_regularization(self):
        parameters = [
            {
                'W': numpy.array([[ 0.35122995, -0.09776762],[ 0.45798496,  1.07694474]]), 
                'b': numpy.array([[0.],[0.]])
            },
            {
                'W': numpy.array([[-0.16557144, -0.16555983],[ 1.11667209,  0.5426583 ]]), 
                'b': numpy.array([[0.],[0.]])
            },
            {
                'W': numpy.array([[-0.33196852,  0.38364789]]), 
                'b': numpy.array([[0.]])
            }
        ]
        regularization = CL.cost_regularization(10, 3, parameters, 0.5)
        print(regularization)

    def test_softmax(self):
        z = numpy.array([[1,2,3.1], [4,-5,0]])
        a = numpy.round(CL.softmax(z), 5)
        expected_a = numpy.round(numpy.array([[0.0474258732, 0.999088949, 0.956892745], [0.952574127, 0.0009110512, 0.04310725]]),5)
        numpy.testing.assert_array_equal(a, expected_a)
        numpy.testing.assert_array_equal(numpy.sum(a, axis=0, keepdims=True), numpy.ones((1,3)))
        
    def test_one_hot_matrix(self):
        labels = numpy.array([[0,1,2,3,1]])
        one_hot_matrix = CL.one_hot_matrix(labels[0], 4)
        expected_matrix = numpy.array([[1,0,0,0,0], [0,1,0,0,1], [0,0,1,0,0], [0,0,0,1,0]])
        numpy.testing.assert_array_equal(one_hot_matrix, expected_matrix)

if __name__ == '__main__':
    unittest.main()