# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class GradientReversalFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None


revgrad = GradientReversalFunc.apply


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)

class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbones, transformer, encoder, action_dim, num_queries):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            action_dim: action dimension 
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = ["rgb1"]

        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj_robot_state = nn.Linear(16, hidden_dim)
        
        if backbones is not None:
            # Use backbone to extract features
            self.is_feature = False
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.encoder_obs_proj = nn.Linear(
                256*3*3, hidden_dim)  # project feature to embedding
        else:
            self.fea_pos_embed = nn.Embedding(1, hidden_dim)
            self.encoder_obs_proj = nn.Linear(
                6, hidden_dim)  # project obs to embedding
            # self.encoder_obs_proj = nn.Linear(
            #     768, hidden_dim)  # project obs to embedding for mvp
            self.end_feat_extractor = nn.Linear(6, 6)
            self.is_feature = True

        # encoder extra parameters
        # self.latent_dim = 256  # final size of latent z # TODO tune
        self.latent_dim = 32  # final size of latent z # TODO tune
        # extra cls token embedding
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(
            16, hidden_dim)  # project action to embedding
        self.encoder_qpos_proj = nn.Linear(
            16, hidden_dim)  # project qpos to embedding
        # project hidden state to latent std, var
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(
            1+1+1+num_queries, hidden_dim))  # [CLS], obs, qpos, a_seq

        # decoder extra parameters
        # project latent sample to embedding
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        # learned position embedding for proprio and latent
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)


    def forward(self, image, qpos, actions=None, is_pad=None):
        """
        image: batch, num_cam, channel, height, width
        qpos: batch, qpos_dim
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape

        if not self.is_feature:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](
                    image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)  # (bs, 256, 7, 7)
            pos = torch.cat(all_cam_pos, axis=3)

        else:
            # Features and position embeddings
    
            point_feat_1 = self.end_feat_extractor(image.reshape(bs,-1)).unsqueeze(-1)
            point_feat_2 = self.end_feat_extractor(torch.cat([image.reshape(bs,-1)[3:],\
                                                              image.reshape(bs,-1)[:3]],dim=0)).unsqueeze(-1)
            point_feat = torch.cat([point_feat_1, point_feat_2], dim=-1)
            point_feat = torch.max(point_feat, dim=-1)[0]
            src = self.encoder_obs_proj(point_feat)  # hidden_dim
            pos = self.fea_pos_embed.weight  # (1, hidden_dim)

        # Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(
                actions)  # (bs, seq, hidden_dim)
            if not self.is_feature:
                feature = F.max_pool2d(src, kernel_size=2, stride=2)
                feature = feature.flatten(1)  # (bs, 256*3*3)
                obs_embed = self.encoder_obs_proj(feature)
            else:
                obs_embed = src
            obs_embed = torch.unsqueeze(obs_embed, axis=1)
            qpos_embed = self.encoder_qpos_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(
                qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1)  # (bs, 1, hidden_dim)
            # (bs, seq+1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, obs_embed, qpos_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(
                1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 3), False).to(
                qpos.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad],
                               axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros(
                [bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        hs = self.transformer(src, None, self.query_embed.weight, pos,
                              latent_input, proprio_input, self.additional_pos_embed.weight, self.is_feature)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)


        return a_hat, is_pad_hat, [mu, logvar]


def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(
        encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build_ACT_model(args):
    action_dim = 16  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    
    # backbones = []
    # backbone = build_backbone(args)
    # backbones.append(backbone)

    # model = DETRVAE(
    #     backbones=backbones,
    #     transformer=transformer,
    #     encoder=encoder,
    #     action_dim=action_dim,
    #     num_queries=args.num_queries,
    #     dann=args.dann
    # )
 
    model = DETRVAE(
        backbones=None,
        transformer=transformer,
        encoder=encoder,
        action_dim=action_dim,
        num_queries=args.num_queries,
    )

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
