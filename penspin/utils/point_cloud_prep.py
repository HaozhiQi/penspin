# --------------------------------------------------------
# Lessons from Learning to Spin “Pens”
# Written by Paper Authors
# Copyright (c) 2024 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# prepare point cloud files
import numpy as np


def sample_cylinder(h, num_points=100, num_circle_points=15, side_points=70):
    # TODO: now we assume the radius of the cylinder is 1
    assert num_points == num_circle_points * 2 + side_points
    pcs = np.zeros((num_points, 3))
    # sample 100 points from top and bottom surface
    r = np.sqrt(np.random.random(num_circle_points))
    theta = np.random.random(num_circle_points) * 2 * np.pi
    pcs[:num_circle_points, 0] = r * np.cos(theta) * 0.5
    pcs[:num_circle_points, 1] = r * np.sin(theta) * 0.5
    pcs[:num_circle_points, 2] = 0.5 * h
    r = np.sqrt(np.random.random(num_circle_points))
    theta = np.random.random(num_circle_points) * 2 * np.pi
    pcs[num_circle_points:num_circle_points * 2, 0] = r * np.cos(theta) * 0.5
    pcs[num_circle_points:num_circle_points * 2, 1] = r * np.sin(theta) * 0.5
    pcs[num_circle_points:num_circle_points * 2, 2] = -0.5 * h
    # sample 400 points from the side surface
    vec = np.random.random((side_points, 2)) - 0.5
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    vec *= 0.5
    pcs[num_circle_points * 2:, :2] = vec
    pcs[num_circle_points * 2:, 2] = h * (np.random.random(side_points) - 0.5)
    return pcs


def sample_cuboid(s_x, s_y, s_z, num_points=100):
    # this function makes a few assumptions: center at (0, 0, 0)
    # side length is s_x, s_y, s_z
    pcs = np.zeros((num_points, 3))
    # assign number of points for each side because it may not divides 6
    idx = np.random.randint(0, 6, size=(num_points,))

    num_points_0 = sum(idx == 0)
    xs = np.random.uniform(0, s_x, size=(num_points_0, 1))
    ys = np.random.uniform(0, s_y, size=(num_points_0, 1))
    zs = np.zeros((num_points_0, 1))
    pcs_0 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_1 = sum(idx == 1)
    xs = np.random.uniform(0, s_x, size=(num_points_1, 1))
    ys = np.random.uniform(0, s_y, size=(num_points_1, 1))
    zs = np.ones((num_points_1, 1)) * s_z
    pcs_1 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_2 = sum(idx == 2)
    xs = np.random.uniform(0, s_x, size=(num_points_2, 1))
    ys = np.zeros((num_points_2, 1))
    zs = np.random.uniform(0, s_z, size=(num_points_2, 1))
    pcs_2 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_3 = sum(idx == 3)
    xs = np.random.uniform(0, s_x, size=(num_points_3, 1))
    ys = np.ones((num_points_3, 1)) * s_z
    zs = np.random.uniform(0, s_z, size=(num_points_3, 1))
    pcs_3 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_4 = sum(idx == 4)
    xs = np.zeros((num_points_4, 1))
    ys = np.random.uniform(0, s_y, size=(num_points_4, 1))
    zs = np.random.uniform(0, s_z, size=(num_points_4, 1))
    pcs_4 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_5 = sum(idx == 5)
    xs = np.ones((num_points_5, 1)) * s_x
    ys = np.random.uniform(0, s_y, size=(num_points_5, 1))
    zs = np.random.uniform(0, s_z, size=(num_points_5, 1))
    pcs_5 = np.concatenate([xs, ys, zs], axis=-1)

    pcs[idx == 0] = pcs_0
    pcs[idx == 1] = pcs_1
    pcs[idx == 2] = pcs_2
    pcs[idx == 3] = pcs_3
    pcs[idx == 4] = pcs_4
    pcs[idx == 5] = pcs_5

    pcs[:, 0] -= s_x / 2
    pcs[:, 1] -= s_y / 2
    pcs[:, 2] -= s_z / 2

    return pcs
