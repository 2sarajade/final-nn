# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    grp1 = [seqs[i] for i in range(len(seqs)) if labels[i]]
    grp2 = [seqs[i] for i in range(len(seqs)) if  not labels[i]]

    class_size = max(len(grp1), len(grp2))
    select1 = list(np.random.choice(grp1, size = class_size, replace = True))
    select2 = list(np.random.choice(grp2, size = class_size, replace = True))
    labels = [True] * class_size + [False] * class_size
    samps = select1 + select2
    combined = list(zip(samps, labels))
    np.random.shuffle(combined)
    shuff_seq, shuff_label = zip(*combined)
    return (shuff_seq, shuff_label)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    translated = []
    for seq in seq_arr:
        translated.append(_one_hot_encode(seq))
    return np.array(translated)

def _one_hot_encode(seq):
    one_hot = {'A':[1,0,0,0],
               'T':[0,1,0,0],
               'C':[0,0,1,0],
               'G':[0,0,0,1]}

    translated = []
    for base in seq:
        translated = translated + one_hot.get(base)
    
    return translated