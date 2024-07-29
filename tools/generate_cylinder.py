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
    output_name,
    ratio_length_diameter,
    radius=0.04,
):
    urdf_base = copy.deepcopy(urdf_base)
    priv_info = []
    priv_info.append([ratio_length_diameter, radius, 1.0])
    for t in ['collision', 'visual']:
        urdf_base.findall(f'.//{t}/geometry/cylinder')[0].attrib['length'] = list2str([radius * 2 * ratio_length_diameter])
        urdf_base.findall(f'.//{t}/geometry/cylinder')[0].attrib['radius'] = list2str([radius])
    urdf_data = ET.tostring(urdf_base)
    with open(output_name, 'wb') as f:
        f.write(urdf_data)
    np.save(f'{output_name.replace(".urdf", ".npy")}', np.array(priv_info))


def main():
    args = arg_parse()
    # load template urdf

    urdf_base = read_xml('assets/cylinder.urdf')
    output_name = f'assets/cylinder/{args.output}'
    os.makedirs(output_name, exist_ok=True)

    # hora 1 x 1 - 1 x 4: [ )
    i = 0
    radius = 0.035
    for s1 in np.arange(0.8, 1.201, 0.1):
        print(s1)
        generate_single_urdf(
            urdf_base,
            os.path.join(output_name, f'{i:04d}.urdf'),
            s1,
            radius
        )
        i += 1
    
    # i = 0
    # for s1 in np.arange(0.8, 1.201, 0.1):
    #     generate_single_urdf(
    #         urdf_base, s1,
    #         os.path.join(output_name, f'{i:04d}.urdf'),
    #     )
    #     i += 1

    # pencil
    # for pencil experiments, we have 8cm diameter [radius 4cm] and 48cm length
    # so scale should be 6x
    # for i, length in enumerate(np.arange(8.0, 10.01, 0.5)):
    #     print(length)
    #     generate_single_urdf(
    #         urdf_base, length,
    #         os.path.join(output_name, f'{i:04d}.urdf'),
    #     )

    # # thin
    # for i, length in enumerate(np.arange(0.16, 0.261, 0.02)):
    #     print(length)
    #     generate_single_urdf(
    #         urdf_base, length,
    #         os.path.join(output_name, f'{i:04d}.urdf'),
    #     )


if __name__ == '__main__':
    main()
