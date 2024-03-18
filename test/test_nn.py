# TODO: import dependencies and write unit tests below
import pytest
import nn.preprocess as preprocess
#import nn.io as io
import nn.nn as nn
import numpy as np

def test_single_forward():
    pass

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

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
    true_dA = np.array([.1, -.1])
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
