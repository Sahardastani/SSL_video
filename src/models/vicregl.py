# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
torch.cuda.empty_cache()
from src.models.feature_extractors.r2p1d import OurVideoResNet
from src.utils import model_utils

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

        # if self.cfg.MODEL.ALPHA < 1.0:
        #     self.maps_projector = model_utils.MLP(cfg.MODEL.MAPS_MLP, self.representation_dim, norm_layer)

        if self.cfg.MODEL.ALPHA > 0.0:
            self.projector = model_utils.MLP(cfg.MODEL.MLP, self.representation_dim, norm_layer)

        self.classifier = nn.Linear(self.representation_dim, self.cfg.DATA.NUM_CLASSES)

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
            self, maps_1, maps_2, location_1, location_2, index_location_1, index_location_2, org_maps_1, org_maps_2
    ):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # L2 distance based bacthing
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

        if self.cfg.MODEL.FAST_VC_REG:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        # # Location based matching
        inv = []
        var = []
        cov = []

        for j in range(index_location_1.shape[0]): # batches
            indexes1= []
            for value in index_location_2[j]:
                closest_index = min(range(len(index_location_1[j])), key=lambda i: abs(index_location_1[j][i] - value))
                indexes1.append(closest_index)
            
            current_index_list = indexes1
            map1_transform = org_maps_1[current_index_list]

            indexes2= []
            for value in index_location_1[j]:
                closest_index = min(range(len(index_location_2[j])), key=lambda i: abs(index_location_2[j][i] - value))
                indexes2.append(closest_index)

            current_index_list = indexes2
            map2_transform = org_maps_2[current_index_list]

            if self.cfg.MODEL.FAST_VC_REG:
                inv_loss_1 = F.mse_loss(map1_transform, org_maps_2)
                inv_loss_2 = F.mse_loss(map2_transform, org_maps_1)
                inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)
            else:
                inv_frame = []
                var_frame = []
                cov_frame = []
                for frame in range(map1_transform.shape[0]):
                    inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(map1_transform[frame], org_maps_2[frame])
                    inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(map2_transform[frame], org_maps_1[frame])
                    inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)
                    var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
                    cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)
                    
                    inv_frame.append(inv_loss)
                    var_frame.append(var_loss)
                    cov_frame.append(cov_loss)

                inv_loss = sum(inv_frame)
                var_loss = sum(var_frame)
                cov_loss = sum(cov_frame)

            inv.append(inv_loss)
            var.append(var_loss)
            cov.append(cov_loss)

        inv_loss = sum(inv)
        var_loss = sum(var)
        cov_loss = sum(cov)

        return inv_loss, var_loss, cov_loss

    def local_loss(self, maps_embedding, locations, index_locations): 
        loss = {
            "inv_loss": [],
            "var_loss": [],
            "cov_loss": []
        }
        for k in range(maps_embedding[0].shape[0]):
            each_frame_maps_embedding = [maps_embedding[0][k], maps_embedding[1][k]]
            num_views = len(maps_embedding)
            inv_loss = 0.0
            var_loss = 0.0
            cov_loss = 0.0
            iter_ = 0
            for i in range(2):
                for j in np.delete(np.arange(np.sum(num_views)), i):
                    inv_loss_this, var_loss_this, cov_loss_this = self._local_loss(
                        each_frame_maps_embedding[i], each_frame_maps_embedding[j], locations[i], locations[j], index_locations[i], index_locations[j], maps_embedding[0], maps_embedding[1]
                    )
                    inv_loss = inv_loss + inv_loss_this
                    var_loss = var_loss + var_loss_this
                    cov_loss = cov_loss + cov_loss_this
                    iter_ += 1

            if self.cfg.MODEL.FAST_VC_REG:
                inv_loss = self.cfg.MODEL.INV_COEFF * inv_loss / iter_
                var_loss = 0.0
                cov_loss = 0.0
                iter_ = 0
                for i in range(num_views):
                    x = model_utils.gather_center(each_frame_maps_embedding[i])
                    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
                    var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
                    x = x.permute(1, 0, 2)
                    *_, sample_size, num_channels = x.shape
                    non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
                    x = x - x.mean(dim=-2, keepdim=True)
                    cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
                    cov_loss = cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels
                    cov_loss = cov_loss + cov_loss.mean()
                    iter_ = iter_ + 1
                var_loss = self.cfg.MODEL.VAR_COEFF * var_loss / iter_
                cov_loss = self.cfg.MODEL.COV_COEFF * cov_loss / iter_
            else:
                inv_loss = inv_loss / iter_
                var_loss = var_loss / iter_
                cov_loss = cov_loss / iter_
            
            loss["inv_loss"].append(inv_loss)
            loss["var_loss"].append(var_loss)
            loss["cov_loss"].append(cov_loss)

        return sum(loss["inv_loss"]), sum(loss["var_loss"]), sum(loss["cov_loss"])

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

    def compute_metrics(self, outputs, is_val= False):
        def correlation_metric(x):
            x_centered = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-05)
            return torch.mean(
                model_utils.off_diagonal((x_centered.T @ x_centered) / (x.size(0) - 1))
            )

        def std_metric(x):
            x = F.normalize(x, p=2, dim=1)
            return torch.mean(x.std(dim=0))

        representation = model_utils.batch_all_gather(outputs["representation"][0])
        corr = correlation_metric(representation)
        stdrepr = std_metric(representation)

        if self.cfg.MODEL.ALPHA > 0.0:
            embedding = model_utils.batch_all_gather(outputs["embedding"][0])
            core = correlation_metric(embedding)
            stdemb = std_metric(embedding)
            if is_val:
                self.log('eval_stdr', stdrepr)
                self.log('eval_stde', stdemb)
                self.log('eval_corr', corr)
                self.log('eval_core', core)
                # return dict(eval_stdr=stdrepr, eval_stde=stdemb, eval_corr=corr, eval_core=core)
            else:
                self.log('train_stdr', stdrepr)
                self.log('train_stde', stdemb)
                self.log('train_corr', corr)
                self.log('train_core', core)
                # return dict(train_stdr=stdrepr, train_stde=stdemb, train_corr=corr, train_core=core)

        return dict(stdr=stdrepr, corr=corr)

    def forward_networks(self, inputs, is_val):
        outputs = {
            "representation": [],
            "embedding": [],
            "layer_1": [],
            "layer_2": [],
            "layer_3": [],
            "layer_4": [],
            "frames_order_layer1": [],
            "frames_order_layer2": [],
            "frames_order_layer3": [],
            "frames_order_layer4": [],
            "index_layer1": [],
            "index_layer2": [],
            "index_layer3": [],
            "index_layer4": [],
            "logits": [],
            "logits_val": [],
        }
        
        # finding out the index of frames in the original video after each layer in resnet
        conv_info = self.backbone.extract_conv_info()
        for x in inputs[1][0:2]:
            current_input = x
            outputs["index_layer1"].append(x)
            for layer, info in conv_info.items():
                for iter, (kernel_size, stride, padding) in enumerate(info):
                    kernel = torch.ones(1, 1, kernel_size).to("cuda")

                    output_tensors = []
                    for row in current_input:
                        row = row.view(1, 1, -1)
                        padded_row = torch.cat([row[:, :, 0:1].repeat(1, 1, padding), row, row[:, :, -1:].repeat(1, 1, padding)], dim=2).to(dtype=kernel.dtype)
                        output_row = F.conv1d(padded_row, kernel, stride=stride)
                        output_row = output_row.view(-1) / kernel_size
                        output_tensors.append(torch.floor(output_row))

                    current_input = torch.stack(output_tensors)
                    
                    if layer == 'layer2' and iter == len(conv_info['layer2'])-1:
                        outputs["index_layer2"].append(current_input)

                    if layer == 'layer3' and iter == len(conv_info['layer3'])-1:
                        outputs["index_layer3"].append(current_input)

                    if layer == 'layer4' and iter == len(conv_info['layer4'])-1:
                        outputs["index_layer4"].append(current_input)

        for x in inputs[0][0:2]:
            out = self.backbone(x)
            layers = [out.layer1_out, out.layer2_out, out.layer3_out, out.layer4_out]
            representation_dim = [64, 128, 256, 512]

            maps = out.layer4_out.flatten(start_dim=2, end_dim=4).permute(0,2,1)
            representation = out.layer_pool_out.view(-1, representation_dim[-1])
            outputs["representation"].append(representation)

            if self.cfg.MODEL.ALPHA > 0.0:
                embedding = self.projector(representation)
                outputs["embedding"].append(embedding)

            if self.cfg.MODEL.ALPHA < 1.0:
                pool = nn.AdaptiveAvgPool2d((1,1))

                for index, layer in enumerate(layers):
                    layer = layer.permute(2,0,1,3,4) #torch.Size([8, 6, 64, 56, 56])
                    pooled_layer = pool(layer) #torch.Size([8, 6, 64, 1, 1])
                    flattened_tensors = pooled_layer.flatten(start_dim=3, end_dim=4).permute(0,1,3,2) #torch.Size([8, 6, 1, 64])
                    num_frames, batch_size, num_loc, embedding_dim = flattened_tensors.shape #torch.Size([8, 6, 1, 64])
                    flattened_tensors_reshaped = flattened_tensors.view(-1, embedding_dim) #torch.Size([48, 64])

                    MAPS_MLP = f'{embedding_dim}-{embedding_dim}-{embedding_dim}'
                    maps_projector = model_utils.MLP(MAPS_MLP, representation_dim[index], norm_layer = "batch_norm").to(device)
                    maps_embedding = maps_projector(flattened_tensors_reshaped) #torch.Size([48, 64])
                    maps_embedding = maps_embedding.view(num_frames, batch_size, num_loc, embedding_dim) #torch.Size([8, 6, 1, 64])
                    outputs[f"layer_{index+1}"].append(maps_embedding)

                    # order of frames in each layer
                    outputs[f"frames_order_layer{index+1}"].append(torch.arange(num_frames).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, -1))

        return outputs

    def forward(self, inputs, is_val=False, backbone_only=False):
        outputs = self.forward_networks(inputs, is_val)
        with torch.no_grad():
            self.compute_metrics(outputs, is_val)
        loss = 0.0

        # Global criterion
        if self.cfg.MODEL.ALPHA > 0.0:
            inv_loss, var_loss, cov_loss = self.global_loss(
                outputs["embedding"]
            )
            loss = loss + self.cfg.MODEL.ALPHA * (inv_loss + var_loss + cov_loss)
            if is_val:
                self.log('eval_inv_l', inv_loss)
                self.log('eval_var_l', var_loss)
                self.log('eval_cov_l', cov_loss)
                self.log('eval_loss', loss)
                # log.update(dict(eval_inv_l=inv_loss, eval_var_l=var_loss, eval_cov_l=cov_loss, eval_loss=loss))
            else:
                self.log('train_inv_l', inv_loss)
                self.log('train_var_l', var_loss)
                self.log('train_cov_l', cov_loss)
                self.log('train_loss', loss)
                # log.update(dict(train_inv_l=inv_loss, train_var_l=var_loss, train_cov_l=cov_loss, train_loss=loss))
            
        # Local criterion
        if self.cfg.MODEL.ALPHA < 1.0:
            (maps_inv_loss_layer1, maps_var_loss_layer1, maps_cov_loss_layer1) = self.local_loss(outputs["layer_1"], outputs["frames_order_layer1"], outputs["index_layer1"]) 
            (maps_inv_loss_layer2, maps_var_loss_layer2, maps_cov_loss_layer2) = self.local_loss(outputs["layer_2"], outputs["frames_order_layer2"], outputs["index_layer2"]) 
            (maps_inv_loss_layer3, maps_var_loss_layer3, maps_cov_loss_layer3) = self.local_loss(outputs["layer_3"], outputs["frames_order_layer3"], outputs["index_layer3"])
            (maps_inv_loss_layer4, maps_var_loss_layer4, maps_cov_loss_layer4) = self.local_loss(outputs["layer_4"], outputs["frames_order_layer4"], outputs["index_layer4"])

            maps_inv_loss = maps_inv_loss_layer1 + maps_inv_loss_layer2 + maps_inv_loss_layer3 + maps_inv_loss_layer4
            maps_var_loss = maps_var_loss_layer1 + maps_var_loss_layer2 + maps_var_loss_layer3 + maps_var_loss_layer4
            maps_cov_loss = maps_cov_loss_layer1 + maps_cov_loss_layer2 + maps_cov_loss_layer3 + maps_cov_loss_layer4

            loss = loss + (1 - self.cfg.MODEL.ALPHA) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )
            self.log('minv_l', maps_inv_loss)
            self.log('mvar_l', maps_var_loss)
            self.log('mcov_l', maps_cov_loss)

        return loss
    
    def training_step(self, train_batch, batch_idx):
        x = train_batch
        loss = self.forward(x)
        return loss
    
    def configure_optimizers(self):
        if self.cfg.MODEL.OPTIMIZER == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
            return optimizer
        elif self.cfg.MODEL.OPTIMIZER == "adam":
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


def neirest_neighbores_on_location(
        input_location, candidate_location, input_maps, candidate_maps, num_matches
):
    """
    input_location: (B, H * W, 2)
    candidate_location: (B, H * W, 2)
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_location.float(), candidate_location.float())
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


def exclude_bias_and_norm(p):
    return p.ndim == 1
