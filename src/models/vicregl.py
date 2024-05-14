# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from src.models.feature_extractors.r2p1d import OurVideoResNet
from src.utils import model_utils

import argparse
import os
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data

from src.datasets.ucf101 import UCF101
from src.utils.svt import utils

device = torch.device("cuda")

class VICRegL(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = int(cfg.MODEL.MLP.split("-")[-1])

        self.train_ucf = []
        self.val_ucf = []
        self.first_epoch = True

        self.dataset_train = UCFReturnIndexDataset(cfg=self.cfg, mode="train", num_retries=10)
        self.dataset_val = UCFReturnIndexDataset(cfg=self.cfg, mode="val", num_retries=10)

        self.train_labels = torch.tensor([s for s in self.dataset_train._labels]).long().cuda()
        self.test_labels = torch.tensor([s for s in self.dataset_val._labels]).long().cuda()

        if "resnet" in cfg.MODEL.ARCH:
            self.backbone, self.representation_dim = OurVideoResNet(), 512
            norm_layer = "batch_norm"
        else:
            raise Exception(f"Unsupported backbone {cfg.MODEL.ARCH}.")
        
        if self.cfg.MODEL.ALPHA < 1.0:
            dim = self.cfg.MODEL.DIM

            self.maps_projector1 = model_utils.MLP(f'{dim[0]}-{dim[0]}-{dim[0]}', dim[0], norm_layer = "batch_norm")
            self.maps_projector2 = model_utils.MLP(f'{dim[1]}-{dim[1]}-{dim[1]}', dim[1], norm_layer = "batch_norm")
            self.maps_projector3 = model_utils.MLP(f'{dim[2]}-{dim[2]}-{dim[2]}', dim[2], norm_layer = "batch_norm")

        if self.cfg.MODEL.ALPHA > 0.0:
            self.projector = model_utils.MLP(cfg.MODEL.MLP, self.representation_dim, norm_layer)

    def _vicreg_loss(self, x, y):
        repr_loss = self.cfg.MODEL.INV_COEFF * F.mse_loss(x, y)

        x = model_utils.gather_center(x)
        y = model_utils.gather_center(y)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.cfg.MODEL.VAR_COEFF * (
                torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
                cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean() 
        cov_loss = self.cfg.MODEL.COV_COEFF * cov_loss

        return repr_loss, std_loss, cov_loss

    def _local_loss(
            self, maps_1, maps_2, index_location_1, index_location_2, org_maps_1, org_maps_2
    ):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # L2 distance based macthing

        m1 = org_maps_1.view(org_maps_1.shape[0], -1, org_maps_1.shape[-1])
        m2 = org_maps_2.view(org_maps_2.shape[0], -1, org_maps_2.shape[-1])
        
        maps_1_filtered, maps_1_nn = neirest_neighbores_on_l2(m1, m2, num_matches=None)
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_l2(m2, m1, num_matches=None)
        
        inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
        inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)
        var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
        cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        # Location based matching

        I = torch.arange(index_location_2.shape[0]).long().unsqueeze(-1).repeat(1, index_location_2.size(-1))
        
        distances = torch.cdist(index_location_1.unsqueeze(-1).float(), index_location_2.unsqueeze(-1).float(), p=1)
        current_index_list1 = torch.min(distances, dim=-1)[1]
        map1_transform = org_maps_1.permute(1, 0, 2, 3)[I, current_index_list1].permute(1, 0, 2, 3)

        distances = torch.cdist(index_location_2.unsqueeze(-1).float(), index_location_1.unsqueeze(-1).float(), p=1)
        current_index_list2 = torch.min(distances, dim=-1)[1]
        map2_transform = org_maps_2.permute(1, 0, 2, 3)[I, current_index_list2].permute(1, 0, 2, 3)

        # 0, 1
        first_map1 = map1_transform.permute(1, 0, 2, 3)[0:8].reshape(-1, 1, map1_transform.shape[-1])
        first_map2 = map2_transform.permute(1, 0, 2, 3)[0:8].reshape(-1, 1, map2_transform.shape[-1])

        first_org1 = org_maps_1.permute(1, 0, 2, 3)[0:8].reshape(-1, 1, org_maps_1.shape[-1])
        first_org2 = org_maps_2.permute(1, 0, 2, 3)[0:8].reshape(-1, 1, org_maps_2.shape[-1])

        first_inv_loss_1, first_var_loss_1, first_cov_loss_1 = self._vicreg_loss(first_map1, first_org2)
        first_inv_loss_2, first_var_loss_2, first_cov_loss_2 = self._vicreg_loss(first_map2, first_org1)

        first_inv = first_inv_loss_1 / 2 + first_inv_loss_2 / 2
        first_var = first_var_loss_1/ 2 + first_var_loss_2 / 2
        first_cov = first_cov_loss_1 / 2 + first_cov_loss_2 / 2

        # 1, 0
        second_map1 = map1_transform.permute(1, 0, 2, 3)[8:16].reshape(-1, 1, map1_transform.shape[-1])
        second_map2 = map2_transform.permute(1, 0, 2, 3)[8:16].reshape(-1, 1, map2_transform.shape[-1])
        
        second_org1 = org_maps_1.permute(1, 0, 2, 3)[8:16].reshape(-1, 1, org_maps_1.shape[-1])
        second_org2 = org_maps_2.permute(1, 0, 2, 3)[8:16].reshape(-1, 1, org_maps_2.shape[-1])

        second_inv_loss_1, second_var_loss_1, second_cov_loss_1 = self._vicreg_loss(second_map1, second_org2)
        second_inv_loss_2, second_var_loss_2, second_cov_loss_2 = self._vicreg_loss(second_map2, second_org1)

        second_inv = second_inv_loss_1 / 2 + second_inv_loss_2 / 2
        second_var = second_var_loss_1 / 2 + second_var_loss_2 / 2
        second_cov = second_cov_loss_1 / 2 + second_cov_loss_2 / 2


        inv_loss = inv_loss + (first_inv / 2 + second_inv / 2)
        var_loss = var_loss + (first_var / 2 + second_var / 2)
        cov_loss = cov_loss + (first_cov / 2 + second_cov / 2)

        return inv_loss, var_loss, cov_loss

    def local_loss(self, maps_embedding, index_locations):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        # iter_ = 0
        
        for k in range(maps_embedding[0].shape[0]):
            each_frame_maps_embedding = [maps_embedding[0][k], maps_embedding[1][k]]

            each_frame_maps_embedding_1 = torch.cat([each_frame_maps_embedding[0],  each_frame_maps_embedding[1]], 0)
            each_frame_maps_embedding_2 = torch.cat([each_frame_maps_embedding[1],  each_frame_maps_embedding[0]], 0)

            index_locations_1 = torch.cat([index_locations[0], index_locations[1]], 0)
            index_locations_2 = torch.cat([index_locations[1], index_locations[0]], 0)

            maps_embedding_1 = torch.cat([maps_embedding[0], maps_embedding[1]], 1)
            maps_embedding_2 = torch.cat([maps_embedding[1], maps_embedding[0]], 1)

            inv_loss_this, var_loss_this, cov_loss_this = self._local_loss(
                each_frame_maps_embedding_1, each_frame_maps_embedding_2, 
                index_locations_1, index_locations_2, 
                maps_embedding_1, maps_embedding_2
            )

            inv_loss = inv_loss + inv_loss_this
            var_loss = var_loss + var_loss_this
            cov_loss = cov_loss + cov_loss_this

        return inv_loss, var_loss, cov_loss

    def global_loss(self, embedding, maps=False):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.cfg.MODEL.INV_COEFF * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = model_utils.gather_center(embedding[i])
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + model_utils.off_diagonal(cov_x).pow_(2).sum().div(
                self.embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.cfg.MODEL.VAR_COEFF * var_loss / iter_
        cov_loss = self.cfg.MODEL.COV_COEFF * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def forward_networks(self, inputs, is_val):
        outputs = {
            "representation": [],
            "embedding": [],
            "layer_1": [],
            "layer_2": [],
            "layer_3": [],
            "index_layer1": [],
            "index_layer2": [],
            "index_layer3": [],
        }
        
        # finding out the index of frames in the original video after each layer in resnet
        conv_info = self.backbone.extract_conv_info()
        for x in inputs[1]:
            current_input = x
            outputs["index_layer1"].append(x)
            for layer, info in conv_info.items():
                for iter, (kernel_size, stride, padding) in enumerate(info):
                    kernel = torch.ones(1, 1, kernel_size).to("cuda")
                    
                    row = current_input.view(8, 1, -1)
                    padded_row = torch.cat([row[:, :, 0:1].repeat(1, 1, padding), row, row[:, :, -1:].repeat(1, 1, padding)], dim=2).to(dtype=kernel.dtype)
                    output_row = F.conv1d(padded_row, kernel, stride=stride)
                    output_row = output_row / kernel_size
                    current_input = output_row.squeeze(1)

                    if layer == 'layer2' and iter == len(conv_info['layer2'])-1:
                        outputs["index_layer2"].append(current_input)

                    if layer == 'layer3' and iter == len(conv_info['layer3'])-1:
                        outputs["index_layer3"].append(current_input)

        # creating local features for representation after each layer of encoder
        for x in inputs[0][0:2]:
            out = self.backbone(x)
            layers = [out.layer1_out, out.layer2_out, out.layer3_out]

            representation = out.layer_pool_out.view(-1, out.layer_pool_out.shape[1])
            outputs["representation"].append(representation)

            # Global
            if self.cfg.MODEL.ALPHA > 0.0:
                embedding = self.projector(representation)
                outputs["embedding"].append(embedding)
            # Local
            if self.cfg.MODEL.ALPHA < 1.0:
                pool = nn.AdaptiveAvgPool2d((1,1))

                for index, layer in enumerate(layers):
                    layer = layer.permute(2,0,1,3,4) #torch.Size([8, 8, 64, 56, 56]) frame, bacth_size
                    pooled_layer = pool(layer) #torch.Size([8, 8, 64, 1, 1])
                    flattened_tensors = pooled_layer.flatten(start_dim=3, end_dim=4).permute(0,1,3,2) #torch.Size([8, 8, 1, 64])
                    num_frames, batch_size, num_loc, embedding_dim = flattened_tensors.shape #torch.Size([8, 8, 1, 64])
                    flattened_tensors_reshaped = flattened_tensors.view(-1, embedding_dim) #torch.Size([64, 64])

                    if index == 0:
                        maps_embedding = self.maps_projector1(flattened_tensors_reshaped).to(device) #torch.Size([64, 64])
                    elif index == 1:
                        maps_embedding = self.maps_projector2(flattened_tensors_reshaped).to(device)
                    elif index == 2:
                        maps_embedding = self.maps_projector3(flattened_tensors_reshaped).to(device)
                    
                    maps_embedding = maps_embedding.view(num_frames, batch_size, num_loc, embedding_dim) #torch.Size([8, 8, 1, 64])
                    outputs[f"layer_{index+1}"].append(maps_embedding)

        return outputs

    def forward(self, inputs, is_val=False, backbone_only=False):
        outputs = self.forward_networks(inputs, is_val)
        loss = 0.0

        # Global criterion
        if self.cfg.MODEL.ALPHA > 0.0:
            inv_loss, var_loss, cov_loss = self.global_loss(
                outputs["embedding"]
            )
            loss = loss + self.cfg.MODEL.ALPHA * (inv_loss + var_loss + cov_loss)

            self.log('global_inv_loss', inv_loss)
            self.log('global_var_loss', var_loss)
            self.log('global_cov_loss', cov_loss)
            self.log('global_loss', loss)
            
        # Local criterion
        if self.cfg.MODEL.ALPHA < 1.0:
            (maps_inv_loss_layer1, maps_var_loss_layer1, maps_cov_loss_layer1) = self.local_loss(outputs["layer_1"], outputs["index_layer1"]) 
            (maps_inv_loss_layer2, maps_var_loss_layer2, maps_cov_loss_layer2) = self.local_loss(outputs["layer_2"], outputs["index_layer2"]) 
            (maps_inv_loss_layer3, maps_var_loss_layer3, maps_cov_loss_layer3) = self.local_loss(outputs["layer_3"], outputs["index_layer3"])

            maps_inv_loss = maps_inv_loss_layer1 + maps_inv_loss_layer2 + maps_inv_loss_layer3
            maps_var_loss = maps_var_loss_layer1 + maps_var_loss_layer2 + maps_var_loss_layer3
            maps_cov_loss = maps_cov_loss_layer1 + maps_cov_loss_layer2 + maps_cov_loss_layer3

            loss = loss + (1 - self.cfg.MODEL.ALPHA) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )
            self.log('local_inv_loss', maps_inv_loss)
            self.log('local_var_loss', maps_var_loss)
            self.log('local_cov_loss', maps_cov_loss)
            self.log('local_loss', loss) # its also contain global loss

        return loss

    @torch.no_grad()
    def knn_classifier(self, train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
        top1, top5, total = 0.0, 0.0, 0
        train_features = train_features.t()
        num_test_images, num_chunks = test_labels.shape[0], 100
        imgs_per_chunk = num_test_images // num_chunks
        retrieval_one_hot = torch.zeros(k, num_classes).cuda()
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images
            features = test_features[
                idx : min((idx + imgs_per_chunk), num_test_images), :
            ]
            targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
            batch_size = targets.shape[0]

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(T).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()
            total += targets.size(0)
        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total
        return top1, top5

    def training_step(self, x):
        loss = self.forward(x)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        model = self.backbone.backbone
        model.cuda()
        model.eval()
        
        if dataloader_idx == 0:
            # train
            feats = model(batch[0])
            self.train_ucf.append(feats)

        elif dataloader_idx == 1:
            # val
            feats = model(batch[0])
            self.val_ucf.append(feats)

    def on_validation_epoch_end(self):
        
        if self.first_epoch == True:
            self.first_epoch = False
        else:
            train_features = torch.cat(self.train_ucf)
            test_features = torch.cat(self.val_ucf)
            
            train_features = nn.functional.normalize(train_features, dim=1, p=2)
            test_features = nn.functional.normalize(test_features, dim=1, p=2)
            
            all_train = self.all_gather(train_features).reshape(-1, 400)
            all_val = self.all_gather(test_features).reshape(-1, 400)

            if self.cfg.TESTsvt.use_cuda:
                all_train = all_train.cuda()
                all_val = all_val.cuda()
                train_labels = self.train_labels.cuda()
                test_labels = self.test_labels.cuda()
                
            print("Features are ready!\nStart the k-NN classification.")
            for k in self.cfg.TESTsvt.nb_knn:
                top1, top5 = self.knn_classifier(all_train, train_labels,
                    all_val, test_labels, k, self.cfg.TESTsvt.temperature)
                print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
                self.log('top1', top1, sync_dist=True)
                self.log('top5', top5, sync_dist=True)

        self.train_ucf.clear()
        self.val_ucf.clear()
    
    def configure_optimizers(self):
        if self.cfg.MODEL.OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
            self.log('lr', 3e-4)
            return optimizer
        else:
            print(f"{self.cfg.MODEL.OPTIMIZER} is not in the optimizer list.")

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0)
    
    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)
    
    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )
    
    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )
    
    indices_selected = indices.masked_select(mask)
    indices_selected = indices_selected.reshape(batch_size, num_matches)

    filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )
    return filtered_input_maps, filtered_candidate_maps

def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_maps, candidate_maps)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)

class UCFReturnIndexDataset(UCF101):
    def __getitem__(self, idx):
        img, _, _, _ = super(UCFReturnIndexDataset, self).__getitem__(idx)
        return img, idx