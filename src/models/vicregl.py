# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.append('/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/src')

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

from src.models.feature_extractors.r2p1d import R2Plus1DNet
from src.utils import model_utils

class VICRegL(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = int(cfg.MODEL.MLP.split("-")[-1])

        if "resnet" in cfg.MODEL.ARCH:
            self.backbone, self.representation_dim = R2Plus1DNet(cfg.MODEL.LAYER_SIZES), 512
            norm_layer = "batch_norm"
        else:
            raise Exception(f"Unsupported backbone {cfg.MODEL.ARCH}.")

        if self.cfg.MODEL.ALPHA < 1.0:
            self.maps_projector = model_utils.MLP(cfg.MODEL.MAPS_MLP, self.representation_dim,
                                                  norm_layer)

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
            self, maps_1, maps_2, location_1, location_2
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

        # Location based matching
        location_1 = location_1.flatten(1, 2)
        location_2 = location_2.flatten(1, 2)

        maps_1_filtered, maps_1_nn = neirest_neighbores_on_location(
            location_1,
            location_2,
            maps_1,
            maps_2,
            num_matches=self.cfg.MODEL.NUM_MATCHES[0],
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_location(
            location_2,
            location_1,
            maps_2,
            maps_1,
            num_matches=self.cfg.MODEL.NUM_MATCHES[1],
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

        return inv_loss, var_loss, cov_loss

    def local_loss(self, maps_embedding, locations):
        num_views = len(maps_embedding)
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss_this, var_loss_this, cov_loss_this = self._local_loss(
                    maps_embedding[i], maps_embedding[j], locations[i], locations[j],
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
                x = model_utils.gather_center(maps_embedding[i])
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
                return dict(eval_stdr=stdrepr, eval_stde=stdemb, eval_corr=corr, eval_core=core)
            else:
                return dict(train_stdr=stdrepr, train_stde=stdemb, train_corr=corr, train_core=core)

        return dict(stdr=stdrepr, corr=corr)

    def forward_networks(self, inputs, is_val):
        outputs = {
            "representation": [],
            "embedding": [],
            "maps_embedding": [],
            "logits": [],
            "logits_val": [],
        }
        for x in inputs:
            representation, list_x1_x5_and_xpool = self.backbone(x)
            outputs["representation"].append(representation)

            if self.cfg.MODEL.ALPHA > 0.0:
                embedding = self.projector(representation)
                outputs["embedding"].append(embedding)

        #     if self.cfg.MODEL.ALPHA < 1.0:
        #         batch_size, num_loc, _ = maps.shape
        #         maps_embedding = self.maps_projector(maps.flatten(start_dim=0, end_dim=1))
        #         maps_embedding = maps_embedding.view(batch_size, num_loc, -1)
        #         outputs["maps_embedding"].append(maps_embedding)

        #     logits = self.classifier(representation.detach())
        #     outputs["logits"].append(logits)

        # if is_val:
        #     _, representation = self.backbone(inputs["val_view"])
        #     val_logits = self.classifier(representation.detach())
        #     outputs["logits_val"].append(val_logits)

        return outputs

    def forward(self, inputs, is_val=False, backbone_only=False):
        # if backbone_only:
        #     maps, _ = self.backbone(inputs)
        #     return maps

        outputs = self.forward_networks(inputs, is_val)
        with torch.no_grad():
            log = self.compute_metrics(outputs, is_val)
        loss = 0.0

        # Global criterion
        if self.cfg.MODEL.ALPHA > 0.0:
            inv_loss, var_loss, cov_loss = self.global_loss(
                outputs["embedding"]
            )
            loss = loss + self.cfg.MODEL.ALPHA * (inv_loss + var_loss + cov_loss)
            if is_val:
                log.update(dict(eval_inv_l=inv_loss, eval_var_l=var_loss, eval_cov_l=cov_loss, eval_loss=loss))
            else:
                log.update(dict(train_inv_l=inv_loss, train_var_l=var_loss, train_cov_l=cov_loss, train_loss=loss))
            
        # Local criterion
        # Maps shape: B, C, H, W
        # With convnext actual maps shape is: B, H * W, C
        # if self.cfg.MODEL.ALPHA < 1.0:
        #     (
        #         maps_inv_loss,
        #         maps_var_loss,
        #         maps_cov_loss,
        #     ) = self.local_loss(
        #         outputs["maps_embedding"], inputs["locations"]
        #     )
        #     loss = loss + (1 - self.cfg.MODEL.ALPHA) * (
        #         maps_inv_loss + maps_var_loss + maps_cov_loss
        #     )
        #     log.update(
        #         dict(minv_l=maps_inv_loss, mvar_l=maps_var_loss, mcov_l=maps_cov_loss,)
        #     )

        # # Online classification

        # labels = inputs["labels"]
        # classif_loss = F.cross_entropy(outputs["logits"][0], labels)
        # acc1, acc5 = model_utils.accuracy(outputs["logits"][0], labels, topk=(1, 5))
        # loss = loss + classif_loss
        # log.update(dict(cls_l=classif_loss, top1=acc1, top5=acc5, l=loss))
        # if is_val:
        #     classif_loss_val = F.cross_entropy(outputs["logits_val"][0], labels)
        #     acc1_val, acc5_val = model_utils.accuracy(
        #         outputs["logits_val"][0], labels, topk=(1, 5)
        #     )
        #     log.update(
        #         dict(clsl_val=classif_loss_val, top1_val=acc1_val, top5_val=acc5_val,)
        #     )

        return loss, log

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        loss, log = self.forward(x[0])
        return loss
    
    # def validation_step(self, val_batch, batch_idx):
    #     x = val_batch
    #     loss, log = self.forward(x[0])
    #     return loss, log 
    
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
    distances = torch.cdist(input_location, candidate_location)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


def exclude_bias_and_norm(p):
    return p.ndim == 1
