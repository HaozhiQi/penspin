import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import open3d as o3d
# import torchsparse
# from torchsparse import SparseTensor
# from torchsparse import nn as spnn
# from torchsparse.nn import functional as F
# from torchsparse.utils.quantize import sparse_quantize

# from functools import partial

# import timm.models.vision_transformer

class SparseCNN(nn.Module):
    def __init__(self, point_channel=7, output_dim=256, voxel_size=0.005):
        super(SparseCNN, self).__init__()
        print(f'SparseCNN')
        self.voxel_size = voxel_size
        self.conv = nn.Sequential(
                    spnn.Conv3d(point_channel, 32, 3),
                    spnn.BatchNorm(32),
                    spnn.ReLU(True),
                    spnn.Conv3d(32, 64, 2, stride=2),
                    spnn.BatchNorm(64),
                    spnn.ReLU(True),
                    spnn.Conv3d(64, 64, 2, stride=2, transposed=True),
                    spnn.BatchNorm(64),
                    spnn.ReLU(True),
                    spnn.Conv3d(64, 32, 3),
                    spnn.BatchNorm(32),
                    spnn.ReLU(True),
                    spnn.Conv3d(32, 10, 1),
                )
    
    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        coords, feats = x[..., :3], x
        print(coords.shape, feats.shape)
        coords -= torch.min(coords, dim=-2, keepdims=True)[0]
        coords = coords.cpu().numpy()
        print(coords.shape)
        coords = [sparse_quantize(coord, self.voxel_size, return_index=True)[0] for coord in coords]
        indices = [sparse_quantize(coord, self.voxel_size, return_index=True)[1] for coord in coords]
        #print(indices[0])
        print(feats.shape)
        feats = [feats[i, indices[i]].cpu() for i in range(len(indices))]
        coords = [torch.tensor(coord, dtype=torch.int).cpu() for coord in coords]
        print(coords[0].shape, feats[0].shape)
        input_sp = [SparseTensor(coords=coords[i], feats=feats[i]) for i in range(len(coords))]
        out = spnn.Conv3d(7, 32, 3)(input_sp[0])
        print(out.feats.shape)
        out = spnn.BatchNorm(32)(out)
        print(out.feats.shape)
        #out = [self.conv(input_sp_tmp).feats for input_sp_tmp in input_sp]
        
        exit(0)


class PointNetAttention(nn.Module):  
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetAttention, self).__init__()

        print(f'PointNetAttention')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp_k = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.local_mlp_v = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.local_mlp_q = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.global_mlp = nn.Linear(1064 * mlp_out_dim, mlp_out_dim)
        self.softmax  = nn.Softmax(dim=-1)
        self.reset_parameters_()
        self.frame_count = 0

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
        q = self.local_mlp_q(x).permute(0, 2, 1)
        k = self.local_mlp_k(x)
        v = self.local_mlp_v(x)
        #print(q.shape, k.shape, v.shape)
        energy = torch.bmm(q, k)  # transpose check
        #print(energy.shape)
        attention = self.softmax(energy).permute(0, 2, 1)
        out = torch.bmm(v, attention)
        #print(out.shape)
        # gloabal max pooling
        # x = torch.max(x, dim=1)[0]
        # out, indices = torch.max(out, dim=1)
        _, indices = torch.max(out, dim=1)
        out = self.global_mlp(out.reshape(out.shape[0], -1))
        return out, indices

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
        self.frame_count = 0

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
        # pc = x[0].cpu().detach().numpy()
        
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        # x = torch.max(x, dim=1)[0]
        x, indices = torch.max(x, dim=1)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # xs = pc[indices[0].cpu().detach().numpy(), 0]
        # ys = pc[indices[0].cpu().detach().numpy(), 1]
        # zs = pc[indices[0].cpu().detach().numpy(), 2]
        # x1 = pc[680:, 0]
        # y1 = pc[680:, 1]
        # z1 = pc[680:, 2]
        # # x1 = fsr_pc[:, 0].cpu()
        # # y1 = fsr_pc[:, 1].cpu()
        # # z1 = fsr_pc[:, 2].cpu()
        # #ax.scatter(xs,ys,zs)
        # ax.scatter(xs, ys, zs, c="red")
        # ax.scatter(x1, y1, z1, c="blue")
        # fig.savefig("tmp/tmp_{}.png".format(str(self.frame_count)))
        # print("save to tmp/tmp_{}.png".format(str(self.frame_count)))
        # self.frame_count += 1
        
        # # save the point cloud and high light the max points
        # # ic(x.shape)
        # # pc = x[0].cpu().detach().numpy()
        # ic(pc.shape)
        # pc = pc.reshape(-1, 3)
        #
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc)
        # # paint the key points red, others blue
        # colors = np.zeros((pc.shape[0], 3))
        # colors[indices[0].cpu().detach().numpy()] = [10, 10, 10]
        #
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        #
        #
        # # view
        # # o3d.visualization.draw_geometries([pcd])
        # o3d.io.write_point_cloud("/home/helin/Code/output/test.ply", pcd)
        #
        # exit()
        return x, indices


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