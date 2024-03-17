# TODO: import dependencies and write unit tests below
import pytest
import nn.preprocess as preprocess
#import nn.io as io
#import nn.nn as nn
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
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

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
