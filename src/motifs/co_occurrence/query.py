from typing import List, cast

import numpy as np
from scipy.spatial import KDTree


def query_tree_range(
        positions_a: List[int],
        positions_b: List[int],
        d_min: int,
        d_max: int,
) -> List[set[int]]:
    positions_a_arr: np.ndarray = np.array(positions_a).reshape(-1, 1)
    positions_b_arr: np.ndarray = np.array(positions_b).reshape(-1, 1)

    # Use KDTree to find neighbors within the specified distance range
    tree_a = KDTree(positions_a_arr)
    tree_b = KDTree(positions_b_arr)

    # Query between two trees
    indices_max = cast(List[List[int]], tree_a.query_ball_tree(tree_b, d_max))
    indices_min = cast(List[List[int]], tree_a.query_ball_tree(tree_b, d_min))
    indices_within = [set(indices_max[i]) - set(indices_min[i]) for i in range(len(indices_max))]

    return indices_within
