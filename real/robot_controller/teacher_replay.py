import cv2
import h5py
import torch
import pickle
import imageio
import argparse

import rospy

from real_robot.robot import *


class MotionReplay:
    def __init__(self, dof_lower, dof_upper, num_actors=1, stack=3, device='cuda'):

        self.num_actors = num_actors
        self.action_scale = 0.04167
        self.actions_num = 16

        self.device = device
        self.visualize = False
        self.states = None
        self.dof_lower = torch.from_numpy(dof_lower).to(self.device)
        self.dof_upper = torch.from_numpy(dof_upper).to(self.device)

        self.current_cmd = 0
        self.num_supported_cmd = 3

        self.need_init = 0

        self.single_step_obs_dim = 64
        self.stack = stack
        self.history = torch.zeros(num_actors, self.single_step_obs_dim * self.stack).float()

        # Data Collection #
        self.save_data = {
            "action": [], "qpos": [], "current_target": [], "tactile": [], "rgb_ori": [],
            "rgb_c2d": [], "depth": [], "pc": [],
        }

        self.cam2hand = np.array([
            [-0.06086204, -0.85466074, -0.51560728,  0.22040379-0.035],              # adjust along the Blue Z axis
            [ 0.90264606, -0.26762265,  0.33705821, -0.05263305-0.038],              # adjust along the Green y axis
            [-0.42605858, -0.44489682,  0.78774421, -0.26671011-0.035],              # adjust along the Red x axis
            [ 0.,          0.,          0.,          1.,             ],
        ])

        gl2cv_homo = np.eye(4)
        self.cam2hand = self.cam2hand @ gl2cv_homo
        self.robot = RealRobot(sam=False,replay=False)

    def deploy(self, seq, save_real_data=True):
        count = 0
        for i in range(len(seq)):
            t0 = time.time()
            cur_pos = self.robot.get_observation()[:16]
            cur_target = seq[i].clone().cpu().numpy()
            rgb_ori, rgb_c2d, depth, pc = self.robot.get_rgb_d()
            # tactile_values, visual_fsrvalue = self.robot.get_tactile()

            index = cur_target[:4]
            thumb = cur_target[4:8]
            mid = cur_target[8:12]
            pinky = cur_target[12:16]

            final_action = [index,mid,pinky,thumb]
            final_action = np.concatenate(final_action)
            delta_action = final_action - cur_pos
            self.robot.set_action(final_action)

            # get fingertip position
            # tip_pos = self.robot.get_tip_pos(obses)
            # tip_pos = torch.from_numpy(tip_pos.astype(np.float32)).cuda()

            # Get the current tactile values
            dt = time.time() - t0
            count += 1
            print(count)
            print("total_fps",1/dt)
            if save_real_data:
                self.save_data["action"].append(delta_action)
                self.save_data["qpos"].append(cur_pos)
                self.save_data["current_target"].append(final_action)
                # self.save_data["tactile"].append(tactile_values)
                self.save_data["rgb_ori"].append(rgb_ori)
                self.save_data["rgb_c2d"].append(rgb_c2d)
                self.save_data["depth"].append(depth)
                self.save_data["pc"].append(pc)

        if save_real_data:
            return self.save_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-collect', action='store_true')
    parser.add_argument('--exp', type=int)
    parser.add_argument('--replay-data-dir', type=str)
    args = parser.parse_args()

    data_collect = args.data_collect
    exp = args.exp

    if data_collect:
        rospy.init_node("hand_grasp")
        replay_data_dir = args.replay_data_dir
        # "./DexPen/outputs/replay_data/teacher_replay_data_1.pkl"
        with open(replay_data_dir, 'rb') as f:
            while True:
                try:
                    sequence = pickle.load(f)
                    seq = sequence['position']
                except:
                    break

        save_real_data = True
        MR = MotionReplay(dof_lower=DOF_LOWER_LIMITS, dof_upper=DOF_UPPER_LIMITS, num_actors=1)
        while True:
            print("init complete")
            time.sleep(3)
            print("init complete")
            real_data = MR.deploy(seq, save_real_data=True)

            if save_real_data:
                with open("./real_data_collection/real_data_{}.pkl".format(exp), 'wb') as f:
                    pickle.dump(real_data, f)
                f.close()

            video_images = real_data["rgb_ori"]
            # this video is for checking the data to slice
            writer = imageio.get_writer('./real_data_collection/real_data_for_check_{}.mp4'.format(exp), fps=1)
            for i in range(len(video_images)):
                img = cv2.cvtColor(video_images[i], cv2.COLOR_BGR2RGB)
                writer.append_data(img)
            writer.close()
    else:
        ### this is for post processing only
        with open("./real_data_collection/real_data_{}.pkl".format(exp), "rb") as f:
            while True:
                try:
                    rd = pickle.load(f)
                except:
                    break

        rd_proccessed = {}
        for key in rd.keys():
            rd_proccessed[key] = rd[key][447:628]

        video_images = rd_proccessed["rgb_ori"]
        writer = imageio.get_writer('./real_data_collection_processed/real_data_{}.mp4'.format(exp), fps=20)

        for i in range(len(video_images)):
            img = cv2.cvtColor(video_images[i], cv2.COLOR_BGR2RGB)
            writer.append_data(img)
        writer.close()

        robot = RealRobot(sam=True,replay=True)
        f = h5py.File('./real_data_collection_processed/real_data.h5', 'a')
        dset = f.create_group("replay_demon_{}".format(exp))
        for key in rd_proccessed.keys():
            dset[key] = rd_proccessed[key]
        obj_ends_list = []
        # annotate_init_frame
        robot.get_point_cloud_with_Tracking_SAM(replay_rgb_c2d=rd_proccessed['rgb_c2d'][0], replay_depth=rd_proccessed['depth'][0], replay_pc=rd_proccessed['pc'][0])
        for i in range(len(rd_proccessed["action"])):
            _ , obj_ends = robot.get_objects_ends(replay_rgb_c2d=rd_proccessed['rgb_c2d'][i], replay_depth=rd_proccessed['depth'][i], replay_pc=rd_proccessed['pc'][i])
            obj_ends_list.append(obj_ends)
        dset["obj_ends"] = obj_ends_list
        f.close()

