# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
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
        if self.cfg.MODEL.L2_ALL_MATCHES:
            num_matches_on_l2 = [None, None]
        else:
            num_matches_on_l2 = self.cfg.MODEL.NUM_MATCHES
        
        maps_1_filtered, maps_1_nn = neirest_neighbores_on_l2(
            maps_1, maps_2, num_matches=num_matches_on_l2[0]
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_l2(
            maps_2, maps_1, num_matches=num_matches_on_l2[1]
        )

        inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
        inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
        var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
        cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        # Location based matching
        
        distances = torch.cdist(index_location_2.unsqueeze(-1).float(), index_location_1.unsqueeze(-1).float(), p=1)
        current_index_list2 = torch.min(distances, dim=-1)[1]
        I = torch.arange(index_location_2.shape[0], ).long().unsqueeze(-1).repeat(1, index_location_2.size(-1))
        map1_transform = org_maps_1.permute(1, 0, 2, 3)[I, current_index_list2].permute(1, 0, 2, 3)

        distances = torch.cdist(index_location_1.unsqueeze(-1).float(), index_location_2.unsqueeze(-1).float(), p=1)
        current_index_list1 = torch.min(distances, dim=-1)[1]
        map2_transform = org_maps_2.permute(1, 0, 2, 3)[I, current_index_list1].permute(1, 0, 2, 3)

        inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(map1_transform.reshape(-1, 1, map1_transform.shape[-1]), org_maps_2.reshape(-1, 1, org_maps_2.shape[-1]))
        inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(map2_transform.reshape(-1, 1, map2_transform.shape[-1]), org_maps_1.reshape(-1, 1, org_maps_1.shape[-1]))

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)
        var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
        cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        return inv_loss, var_loss, cov_loss

    def local_loss(self, maps_embedding, index_locations):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        
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
            iter_ += 1

        inv_loss = inv_loss / iter_
        var_loss = var_loss / iter_
        cov_loss = cov_loss / iter_

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
            self.log('local_loss', loss) # I guess its also contain global loss

        return loss

    def extract_feature_pipeline(self):
        # ============ preparing data ... ============
        dataset_train = UCFReturnIndexDataset(cfg=self.cfg, mode="train", num_retries=10)
        dataset_val = UCFReturnIndexDataset(cfg=self.cfg, mode="val", num_retries=10)

        train_labels = torch.tensor([s for s in dataset_train._labels]).long().cuda()
        test_labels = torch.tensor([s for s in dataset_val._labels]).long().cuda()

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.cfg.TESTsvt.batch_size_per_gpu,
            num_workers=self.cfg.TESTsvt.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.cfg.TESTsvt.batch_size_per_gpu,
            num_workers=self.cfg.TESTsvt.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model = self.backbone.backbone
        model.cuda()
        model.eval()

        # ============ extract features ... ============
        print("Extracting features for train set...")
        train_features = self.extract_features(model, data_loader_train)
        print("Extracting features for val set...")
        test_features = self.extract_features(model, data_loader_val)

        train_features = torch.cat(train_features)
        test_features = torch.cat(test_features)

        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

        return train_features, test_features, train_labels, test_labels

    @torch.no_grad()
    def extract_features(self, model, dataloader):
        features = []
        metric_logger = utils.MetricLogger(delimiter="  ")
        for samples, index in metric_logger.log_every(dataloader, 10):
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            feats = model(samples)
            features.append(feats)
        return features

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

        # # validation step
        # train_features, test_features, train_labels, test_labels = self.extract_feature_pipeline()
        # print("Features are ready!\nStart the k-NN classification.")
        # for k in self.cfg.TESTsvt.nb_knn:
        #     top1, top5 = self.knn_classifier(train_features, train_labels, test_features, test_labels, k, self.cfg.TESTsvt.temperature)
        #     self.log('top1', top1)
        #     self.log('top5', top5)
        #     print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
        return loss
    
    def configure_optimizers(self):
        if self.cfg.MODEL.OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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