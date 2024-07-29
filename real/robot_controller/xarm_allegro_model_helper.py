#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import time
import rospy
import threading
import numpy as np
import sapien.core as sapien
from typing import List
import transforms3d

# from xarm.wrapper import XArmAPI


class XarmAllegro_ModelCalculate:
    def __init__(self, urdf_path):
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.xarm = loader.load(urdf_path)
        self.scene.step()
        self.xarm.set_pose(sapien.Pose())
        self.xarm_model = self.xarm.create_pinocchio_model()
        self.joint_names = [joint.get_name() for joint in self.xarm.get_active_joints()]
        self.link_name2id = {self.xarm.get_links()[i].get_name(): i for i in range(len(self.xarm.get_links()))}
        # print(self.link_name2id)


    # def compute_end_link_spatial_jacobian(self, qpos):
    #     self.xarm.set_qpos(qpos)
    #     jacobian = self.xarm.compute_world_cartesian_jacobian()
    #     return jacobian
    
    #compute special jacobian for the xarm endeffect
    def compute_xarm_endlink_jacobian(self,qpos):
        full_jacobian = self.xarm_model.compute_full_jacobian(qpos)
        link_jacobian = self.xarm_model.get_link_jacobian(6,local = True)
        return link_jacobian
    
    #compute xarm endeffect linkpose from qpos
    def compute_xarmfk_pose(self,qpos,index):
        fk = self.xarm_model.compute_forward_kinematics(qpos)
        link_pose = self.xarm_model.get_link_pose(index)
        return link_pose
    #compute xarm endeffect link cartesian from qpos
    def compute_xarmfk_cartesian(self,qpos):
        fk = self.xarm_model.compute_forward_kinematics(qpos)
        link_pose = self.xarm_model.get_link_pose(7)
        p=link_pose.p*1000
        q=link_pose.q
        eluer = transforms3d.euler.quat2euler(q,axes='sxyz')
        transformation_matrix = link_pose.to_transformation_matrix()
        x,y,z = p
        r,p,yy = eluer
        #return p,q
        return [x,y,z,r*180/np.pi,p*180/np.pi,yy*180/np.pi]

    #compute xarm inverse kinematic
    def compute_xarm_ik(self,pose,initial_qpos,active_qmask):
        ik = self.xarm_model.compute_inverse_kinematics(link_index=7, pose=pose,initial_qpos=initial_qpos,active_qmask=active_qmask)
        return ik
    
    #compute xarm inverse kinematic(qpos) from cartesian of endeffect
    def compute_xarm_ik_from_cartesian(self,cartesian,initial_qpos,active_qmask):
        p = cartesian[0:3]
        q = transforms3d.euler.euler2quat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
        pose = sapien.Pose(p=p, q=q)
        ik_from_cartesian = self.compute_xarm_ik(pose=pose,initial_qpos=initial_qpos,active_qmask= np.array([True,True,True,True,True,True]))
        return ik_from_cartesian

    #compute xarm inverse kinematic(qpos) from TF
    def compute_xarm_ik_from_tf(self, cartesian,initial_qpos,active_qmask):
        return ik_from_TF