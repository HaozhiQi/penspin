import argparse
import copy
import os

import numpy as np
import xml.etree.ElementTree as ET


def arg_parse():
    parser = argparse.ArgumentParser(description='URDF generator')
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def read_xml(filename):
    root = ET.parse(filename).getroot()
    return root


# Convert a list to a space-separated string
def list2str(in_list):
    out = ''
    for el in in_list:
        out += str(el) + ' '
    return out[:-1]


def generate_single_urdf(
    urdf_base,
    s1, s2,
    output_name,
):
    urdf_base = copy.deepcopy(urdf_base)
    priv_info = []
    # scale variation (only the box)
    # 0.01 is some small number to make the cube thin
    # 0.08 is the diameter of the ball
    priv_info.append([s1, s2, 1.0])
    urdf_base.findall(f'.//collision/geometry/box')[0].attrib['size'] = list2str([s1 * 0.08, s2 * 0.08, 0.08])
    urdf_base.findall(f'.//visual/geometry/mesh')[0].attrib['scale'] = list2str([s1 * 0.08, s2 * 0.08, 0.08])
    urdf_data = ET.tostring(urdf_base)
    with open(output_name, 'wb') as f:
        f.write(urdf_data)
    np.save(f'{output_name.replace(".urdf", ".npy")}', np.array(priv_info))


def main():
    args = arg_parse()
    # load template urdf

    urdf_base = read_xml('assets/cube.urdf')
    output_name = f'assets/cuboid/{args.output}'
    os.makedirs(output_name, exist_ok=True)

    i = 0
    # the canonical side length will always be 1, so s1 and s2 is realtive to 1
    for s1 in np.arange(0.9, 1.101, 0.05):
        for s2 in np.arange(0.9, 1.101, 0.05):
            generate_single_urdf(
                urdf_base, s1, s2,
                os.path.join(output_name, f'{i:04d}.urdf'),
            )
            i += 1


if __name__ == '__main__':
    main()