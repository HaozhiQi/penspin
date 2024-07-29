import numpy as np
import math

image_dict = {
    'base_link': 0,
    'palm': 1,
    'wrist': 2,
    'link_0.0': 3,
    'link_1.0': 4,
    'link_1.0_fsr': 5,
    'link_2.0': 6,
    'link_2.0_fsr': 7,
    'link_3.0': 8,
    'link_3.0_tip': 9,
    'link_3.0_tip_fsr_2': 10,
    'link_3.0_tip_fsr_5': 11,
    'link_3.0_tip_fsr_8': 12,
    'link_3.0_tip_fsr_11': 13,
    'link_3.0_tip_fsr_13': 14,
    'link_4.0': 15,
    'link_5.0': 16,
    'link_5.0_fsr': 17,
    'link_6.0': 18,
    'link_6.0_fsr': 19,
    'link_7.0': 20,
    'link_7.0_tip': 21,
    'link_7.0_tip_fsr_2': 22,
    'link_7.0_tip_fsr_5': 23,
    'link_7.0_tip_fsr_8': 24,
    'link_7.0_tip_fsr_11': 25,
    'link_7.0_tip_fsr_13': 26,
    'link_8.0': 27,
    'link_9.0': 28,
    'link_9.0_fsr': 29,
    'link_10.0': 30,
    'link_10.0_fsr': 31,
    'link_11.0': 32,
    'link_11.0_tip': 33,
    'link_11.0_tip_fsr_2': 34,
    'link_11.0_tip_fsr_5': 35,
    'link_11.0_tip_fsr_8': 36,
    'link_11.0_tip_fsr_11': 37,
    'link_11.0_tip_fsr_13': 38,
    'link_12.0': 39,
    'link_13.0': 40,
    'link_14.0': 41,
    'link_14.0_fsr': 42,
    'link_15.0': 43,
    'link_15.0_fsr': 44,
    'link_15.0_tip': 45,
    'link_15.0_tip_fsr_2': 46,
    'link_15.0_tip_fsr_5': 47,
    'link_15.0_tip_fsr_8': 48,
    'link_15.0_tip_fsr_11': 49,
    'link_15.0_tip_fsr_13': 50,
    'link_base_fsr': 51,
    'link_8.0_fsr': 52,
    'link_4.0_fsr': 53,
    'link_0.0_fsr': 54
}

contact_sensor_names = [
    "link_1.0_fsr",
    "link_2.0_fsr",
    "link_3.0_tip_fsr_2",
    "link_3.0_tip_fsr_5",
    "link_3.0_tip_fsr_8",
    "link_3.0_tip_fsr_11",
    "link_3.0_tip_fsr_13",
    "link_5.0_fsr",
    "link_6.0_fsr",
    "link_7.0_tip_fsr_2",
    "link_7.0_tip_fsr_5",
    "link_7.0_tip_fsr_8",
    "link_7.0_tip_fsr_11",
    "link_7.0_tip_fsr_13",
    "link_9.0_fsr",
    "link_10.0_fsr",
    "link_11.0_tip_fsr_2",
    "link_11.0_tip_fsr_5",
    "link_11.0_tip_fsr_8",
    "link_11.0_tip_fsr_11",
    "link_11.0_tip_fsr_13",
    "link_14.0_fsr",
    "link_15.0_fsr",
    "link_15.0_tip_fsr_2",
    "link_15.0_tip_fsr_5",
    "link_15.0_tip_fsr_8",
    "link_15.0_tip_fsr_11",
    "link_15.0_tip_fsr_13",
    "link_base_fsr",
    "link_0.0_fsr",
    "link_4.0_fsr",
    "link_8.0_fsr"

]
STL_FILE_DICT = {
    'base_link': "assets/round_tip/meshes/visual/base_link.obj",
    'link_0.0': "assets/round_tip/meshes/visual/link_0.0.obj",
    'link_1.0': "assets/round_tip/meshes/visual/link_1.0.obj",
    'link_2.0': "assets/round_tip/meshes/visual/link_2.0.obj",
    'link_3.0': "assets/round_tip/meshes/visual/link_3.0.obj",
    'link_3.0_tip': "assets/round_tip/meshes/visual/link_tip.obj",
    'link_4.0': "assets/round_tip/meshes/visual/link_0.0.obj",
    'link_5.0': "assets/round_tip/meshes/visual/link_1.0.obj",
    'link_6.0': "assets/round_tip/meshes/visual/link_2.0.obj",
    'link_7.0': "assets/round_tip/meshes/visual/link_3.0.obj",
    'link_7.0_tip': "assets/round_tip/meshes/visual/link_tip.obj",
    'link_8.0': "assets/round_tip/meshes/visual/link_0.0.obj",
    'link_9.0': "assets/round_tip/meshes/visual/link_1.0.obj",
    'link_10.0': "assets/round_tip/meshes/visual/link_2.0.obj",
    'link_11.0': "assets/round_tip/meshes/visual/link_3.0.obj",
    'link_11.0_tip': "assets/round_tip/meshes/visual/link_tip.obj",
    'link_12.0': "assets/round_tip/meshes/visual/link_12.0_right.obj",
    'link_13.0': "assets/round_tip/meshes/visual/link_13.0.obj",
    'link_14.0': "assets/round_tip/meshes/visual/link_14.0.obj",
    'link_15.0': "assets/round_tip/meshes/visual/link_15.0.obj",
    'link_15.0_tip': "assets/round_tip/meshes/visual/link_tip.obj"
}

DOF_LOWER_LIMITS = np.array([-0.5585, -0.27924, -0.27924, -0.27924,
                             0.27924, -0.331603, -0.27924, -0.27924,
                             -0.5585, -0.27924, -0.27924, -0.27924,
                             -0.5585, -0.27924, -0.27924, -0.27924
                             ])
DOF_UPPER_LIMITS = np.array([0.5585, 1.727825, 1.727825, 1.727825,
                             1.57075, 1.1518833, 1.727825, 1.76273055,
                             0.5585, 1.727825, 1.727825, 1.727825,
                             0.5585, 1.727825, 1.727825, 1.727825
                             ])

INIT_QPOS = np.array([-0.0579, 0.9454, 0.7381, 0.4866,
                      1.2916, 0.5402, 0.7264, 0.3481,
                      0.0331, 1.3008, 0.1790, 0.2358,
                      0.1153, 1.0150, 1.0156, 0.2932], dtype=np.float32)


def nomolize(joint_state):
    return (joint_state - DOF_LOWER_LIMITS) / (DOF_UPPER_LIMITS - DOF_LOWER_LIMITS) / 0.5 - 1


def scale(actions):
    return 0.5 * (actions + 1.0) * (DOF_UPPER_LIMITS - DOF_LOWER_LIMITS) + DOF_LOWER_LIMITS


def fsr_align(fsrValue):
    # change order align to sim( fsrValue is acquired from the hand in the real system, the indices of fsrValues are indicated with pink color in the layout map)
    fsr_value = np.array([fsrValue[20], fsrValue[21], fsrValue[22],  # joint 1, joint 2, joint 5
                          fsrValue[23], fsrValue[24], fsrValue[25],  # joint 6, joint 9, joint 10
                          fsrValue[26], fsrValue[27], fsrValue[28],  # joint 14, joint 15, joint 0
                          # fsrValue[29], fsrValue[30], fsrValue[31],                                              # joint 4, joint 8, joint 13
                          fsrValue[0], fsrValue[2], fsrValue[1], fsrValue[3], fsrValue[5],  # index tip
                          fsrValue[19], fsrValue[18], fsrValue[17], fsrValue[16], fsrValue[7],  # thumb tip
                          fsrValue[14], fsrValue[15], fsrValue[12], fsrValue[13], fsrValue[6],  # middle finger
                          fsrValue[10], fsrValue[8], fsrValue[11], fsrValue[9], fsrValue[4]])  # pinky finger
    return fsr_value


def _action_hora2allegro(actions):
    obs_index = actions[0:4]
    obs_thumb = actions[4:8]
    obs_middle = actions[8:12]
    obs_ring = actions[12:16]
    cmd_act = np.concatenate([obs_index, obs_middle, obs_ring, obs_thumb]).astype(np.float32)
    return cmd_act


def _obs_allegro2hora(obses):
    obs_index = obses[0:4]
    obs_middle = obses[4:8]
    obs_ring = obses[8:12]
    obs_thumb = obses[12:16]
    obses = np.concatenate([obs_index, obs_thumb, obs_middle, obs_ring]).astype(np.float32)
    return obses


def rgba2rgb(rgba, background=(255, 255, 255)):
    h, w, c = rgba.shape

    if c == 3:
        return rgba

    rgb = np.zeros((h, w, 3), dtype='float32')
    r, g, b, a = rgba[..., 2], rgba[..., 1], rgba[..., 0], rgba[..., 3]

    a = np.asanyarray(a, dtype='float32') / 255

    R, G, B = background
    rgb[..., 0] = r * a + (1.0 - a) * R
    rgb[..., 1] = g * a + (1.0 - a) * G
    rgb[..., 2] = b * a + (1.0 - a) * B

    return np.asanyarray(rgb)


def mask2box(mask):
    w = 0
    flag = True
    for i in range(len(mask.shape[0])):
        leng = 0
        for j in range(len(mask.shape[1])):
            if mask[i][j] == 1:
                leng += 1
                if flag:
                    init_x, init_y = i, j
                    flag = False
        if leng > 0:
            w += 1
            l = leng

    points = [(init_x, init_y), (init_x + l - 1, init_y), (init_x, init_y + w - 1), (init_x + l - 1, init_y + w - 1)]

    return points


def get_furthest_point(ref_point, points):
    max_dis = -1
    id = 0
    x_ref, y_ref, z_ref = ref_point

    for i in range(len(points)):
        pc = points[i]
        dis = np.sqrt((pc[0] - x_ref) ** 2 + (pc[1] - y_ref) ** 2 + (pc[2] - z_ref) ** 2)
        if dis >= max_dis:
            id = i
            max_dis = dis

    return id
