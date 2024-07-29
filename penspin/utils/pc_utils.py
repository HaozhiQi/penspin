# --------------------------------------------------------
# Lessons from Learning to Spin “Pens”
# Written by Paper Authors
# Copyright (c) 2024 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import numpy as np
import torch
from typing import List


def batch_index_select(input_tensor, index, dim):
    """Batch index_select
    Code Source: https://github.com/Jiayuan-Gu/torkit3d/blob/master/torkit3d/nn/functional.py
    Args:
        input_tensor (torch.Tensor): [B, ...]
        index (torch.Tensor): [B, N] or [B]
        dim (int): the dimension to index
    References:
        https://discuss.pytorch.org/t/batched-index-select/9115/7
        https://github.com/vacancy/AdvancedIndexing-PyTorch
    """

    if index.dim() == 1:
        index = index.unsqueeze(1)
        squeeze_dim = True
    else:
        assert (
                index.dim() == 2
        ), "index is expected to be 2-dim (or 1-dim), but {} received.".format(
            index.dim()
        )
        squeeze_dim = False
    assert input_tensor.size(0) == index.size(0), "Mismatched batch size: {} vs {}".format(
        input_tensor.size(0), index.size(0)
    )
    views = [1 for _ in range(input_tensor.dim())]
    views[0] = index.size(0)
    views[dim] = index.size(1)
    expand_shape = list(input_tensor.shape)
    expand_shape[dim] = -1
    index = index.view(views).expand(expand_shape)
    out = torch.gather(input_tensor, dim, index)
    if squeeze_dim:
        out = out.squeeze(1)
    return out


def generate_normalized_pc(width, height, intrinsic_mat):
    v, u = np.indices([height, width])
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones([height, width])], axis=-1)
    normalized_pc = uv1 @ np.linalg.inv(intrinsic_mat).T
    return normalized_pc


# @timeit
def process_robot_pc(pc: torch.Tensor, bound: List[float]):
    from pytorch3d import ops
    batch = pc.shape[0]
    pc = pc.reshape([batch, -1, 3])
    # segmentation = segmentation.reshape([batch, -1])
    assert len(bound) == 6
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = within_bound_x & within_bound_y & within_bound_z
    within_indices = [torch.nonzero(within_bound[i]) for i in range(batch)]
    within_indices_len = [len(a) for a in within_indices]
    max_num_points = max(within_indices_len)
    max_num_points = max(max_num_points, max_num_points)
    index_list = []
    for i in range(batch):
        indices = within_indices[i][:, 0]
        num_index = len(indices)
        indices = torch.cat(
            [indices, torch.zeros(max_num_points - num_index, dtype=indices.dtype, device=indices.device)])
        index_list.append(indices)

    batch_indices = torch.stack(index_list)
    batch_pc = batch_index_select(pc, batch_indices, dim=1)
    batch_pc = ops.sample_farthest_points(
        batch_pc, torch.tensor(within_indices_len, device=pc.device, dtype=torch.int), K=512
    )[0]
    return batch_pc