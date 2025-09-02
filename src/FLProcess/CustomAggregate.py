
from functools import reduce
from typing import List, Tuple

import numpy as np

from flwr.common import NDArray, NDArrays

def weightedAggregate(results: List[Tuple[NDArrays, int]], scores:List[int], is_suspicious_scores_added=True) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    if is_suspicious_scores_added:
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        weighted_weights = [[layer * scores[i] for layer in weights] for i, weights in enumerate(weighted_weights)]

    # Create a list of weights, each multiplied by the related number of examples
    else:
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime