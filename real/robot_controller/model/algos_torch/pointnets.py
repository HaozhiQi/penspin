import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d

# from functools import partial

#import timm.models.vision_transformer
class PointNet(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNet, self).__init__()

        print(f'PointNetSmall')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.reset_parameters_()
        self.forward_num = 0

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        #pc = x[0].cpu().detach().numpy()
        #print(x[0].shape)
        # Local
        point = x.clone()
        x = self.local_mlp(x)
        print(x)
        # gloabal max pooling
        x, arg_max = torch.max(x, dim=1)[0:2]
        saved_dict = dict(point=point.cpu().numpy(), arg_max=arg_max.cpu().numpy())
        import pickle
        with open(f"/media/binghao/DATA/new_rl_pipeline/rl_pipeline/robot_controller/pointnet_debug/{self.forward_num}.pkl", "wb") as f:
            pickle.dump(saved_dict, f)
        self.forward_num += 1
        #x, indices = torch.max(x, dim=1)
        # save the point cloud and high light the max points
        #ic(x.shape)
        #pc = x[0].cpu().detach().numpy()
        #pc = x[0]
        #print(pc)
        #ic(pc.shape)
        # pc = pc.reshape(-1, 3)
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc)
        # # paint the key points red, others blue
        # colors = np.zeros((pc.shape[0], 3))
        # colors[indices[0].cpu().detach().numpy()] = [10, 10, 10]
        
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        
        
        # # view
        # # o3d.visualization.draw_geometries([pcd])
        # o3d.io.write_point_cloud("/media/binghao/DATA/new_rl_pipeline/rl_pipeline/pcd/test.ply", pcd)
        
        # exit()
        return x


class PointNetMedium(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetMedium, self).__init__()

        print(f'PointNetMedium')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, mlp_out_dim),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x


class PointNetLarge(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetLarge, self).__init__()

        print(f'PointNetLarge')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, mlp_out_dim),
        )

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x


class PointNetLargeHalf(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetLargeHalf, self).__init__()

        print(f'PointNetLargeHalf')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, mlp_out_dim),
            # nn.GELU(),
            # nn.Linear(128, 256),
            # nn.GELU(),
            # nn.Linear(256, mlp_out_dim),
        )

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x

if __name__ == '__main__':
    b = 2
    img = torch.zeros(size=(b, 64, 64, 3))

    extractor = ResNet50(output_channel=256)

    out = extractor(img)
    print(out.shape)