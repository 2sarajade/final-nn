# TODO: import dependencies and write unit tests below
import pytest
import nn.preprocess as preprocess
import nn.io as io
import nn.nn as nn
import numpy as np

def test_single_forward():
    nnetwork = nn.NeuralNetwork(nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}], 
                                lr = .01, seed = 23, batch_size = 1, epochs = 1, loss_function = "_binary_cross_entropy" )

    W_curr = np.array([[0, 1], [1, -1]])
    b_curr = np.array([[0], [1]])
    A_prev = np.array([[.5, .5]])
    activation = 'relu'
    A_curr, Z_curr = nnetwork._single_forward(W_curr, b_curr, A_prev, activation)
    true_Z_curr = np.array([[.5], [1]])
    true_A_curr = np.array([[.5, 1]])
    assert np.allclose(Z_curr, true_Z_curr)
    assert np.allclose(A_curr, true_A_curr)



def test_forward():
    nnetwork = nn.NeuralNetwork(nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
                                           {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}], 
                                lr = .01, seed = 23, batch_size = 1, epochs = 1, loss_function = "_binary_cross_entropy" )
    X = np.array([[1,0]])
    output, cache = nnetwork.forward(X)
    assert output.shape == (1,1)
    assert cache.get("A2").shape == (1,1)
    assert cache.get("Z2").shape == (1,1)

def test_single_backprop():
    nnetwork = nn.NeuralNetwork(nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}], 
                                lr = .01, seed = 23, batch_size = 1, epochs = 1, loss_function = "_binary_cross_entropy" )

    W_curr = np.array([[0, 1], [1, -1]])
    b_curr = np.array([[0], [1]])
    A_prev = np.array([[1], [-1]])
    dA_curr = np.array([[.5],[.5]])
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    activation = 'relu'

    dA_prev, dW_curr, db_curr = nnetwork._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation)

    true_dA_prev = np.array([[.5, -.5]])
    true_dW_curr = np.array([[0], [0]])
    true_db_curr = np.array([[0],[1]])
    assert np.allclose(dA_prev, true_dA_prev)
    assert np.allclose(dW_curr, true_dW_curr)
    assert np.allclose(db_curr, true_db_curr)

def test_predict():
    nnetwork = nn.NeuralNetwork(nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}], 
                                lr = .01, seed = 23, batch_size = 1, epochs = 1, loss_function = "_binary_cross_entropy" )
    X = np.array([[0,1]])
    predict = nnetwork.predict(X)
    forward = nnetwork.forward(X)[0]
    assert np.allclose(predict, forward)

def test_binary_cross_entropy():
    nnetwork = nn.NeuralNetwork(nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}], 
                                lr = .01, seed = 23, batch_size = 1, epochs = 1, loss_function = "_binary_cross_entropy" )
    y = np.array([1, 0])
    y_hat = np.array([.9, .1])
    my_loss = nnetwork._binary_cross_entropy(y, y_hat)
    true_loss = -np.log(.9)
    
    assert np.allclose(my_loss, true_loss, rtol = 0.000001)

def test_binary_cross_entropy_backprop():
    nnetwork = nn.NeuralNetwork(nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}], 
                                lr = .01, seed = 23, batch_size = 1, epochs = 1, loss_function = "_binary_cross_entropy" )
    y = np.array([1, 0])
    y_hat = np.array([.9, .1])
    true_dA = np.array([-1/1.8, -1/1.8])
    dA = nnetwork._binary_cross_entropy_backprop(y, y_hat)
    assert np.allclose(dA, true_dA, rtol = 0.000001)

def test_mean_squared_error():
    nnetwork = nn.NeuralNetwork(nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}], 
                                lr = .01, seed = 23, batch_size = 1, epochs = 1, loss_function = "_binary_cross_entropy" )
    y = np.array([1, 0])
    y_hat = np.array([.9, .1])
    my_loss = nnetwork._mean_squared_error(y, y_hat)
    true_loss = .01
    
    assert np.allclose(my_loss, true_loss, rtol = 0.000001)

def test_mean_squared_error_backprop():
    nnetwork = nn.NeuralNetwork(nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}], 
                                lr = .01, seed = 23, batch_size = 1, epochs = 1, loss_function = "_binary_cross_entropy" )
    y = np.array([1, 0])
    y_hat = np.array([.9, .1])
    true_dA = np.array([-.1, .1])
    dA = nnetwork._mean_squared_error_backprop(y, y_hat)
    assert np.allclose(dA, true_dA, rtol = 0.000001)

def test_sample_seqs():
    seqs = ["AA", "CC", "AG", "GT", "TC", "CA"]
    labels = [1,1,0,0,0,0]
    sseqs, slabels = preprocess.sample_seqs(seqs, labels)

    assert len(sseqs) == len(slabels)
    assert np.sum(slabels) == 4
    assert len(sseqs) == 8


def test_one_hot_encode_seqs():
    """
    A -> [1, 0, 0, 0]
    T -> [0, 1, 0, 0]
    C -> [0, 0, 1, 0]
    G -> [0, 0, 0, 1]
    """
    seqs = ["ATCG", "AAAA"]
    one_hot = preprocess.one_hot_encode_seqs(seqs)
    truth = np.array([
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    ])

    assert one_hot.shape[0] == 2
    assert one_hot.shape[1] == 16
    assert np.array_equal(one_hot, truth)
