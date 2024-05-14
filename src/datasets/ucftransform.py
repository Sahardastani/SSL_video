# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import random
import warnings
import numpy as np

import hydra
import torch
import torch.utils.data
from einops import rearrange
from logging import getLogger as get_logger

from src.utils.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
from src.datasets.decoder import decode
from src.datasets.transform import VideoDataAugmentationDINO
from src.datasets.video_container import get_video_container
from src.utils.defaults import build_config
logger = get_logger(__name__)

class UCFtransform(torch.utils.data.Dataset):
    """
    UCF101 video loader. Construct the UCF101 video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10, get_flow=False):
        """
        Construct the UCF101 video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for UCF101".format(mode)
        self.mode = mode
        self.cfg = cfg
        if get_flow:
            assert mode == ["train", "val"], "invalid: flow only for train mode"
        self.get_flow = get_flow

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        print("Constructing UCF101 {}...".format(mode))
        self._construct_loader()
        self._path_to_videos = np.asarray(self._path_to_videos)
        self._labels = np.asarray(self._labels)
        self._spatial_temporal_idx = np.asarray(self._spatial_temporal_idx)

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            # self.cfg.DATA.PATH_PREFIX, 'annotations', "{}.csv".format(self.mode)
            self.cfg.DATA.PATH_PREFIX, "ucf101_{}_split_1_videos.txt".format(self.mode)
            # self.cfg.DATA.PATH_PREFIX, "{}.txt".format(self.mode)

        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                        len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                        == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, 'videos', path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
                len(self._path_to_videos) > 0
        ), "Failed to load UCF101 split {} from {}".format(
            self._split_idx, path_to_file
        )
        print(
            "Constructing UCF101 dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                    self._spatial_temporal_idx[index]
                    // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                        self._spatial_temporal_idx[index]
                        % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                     + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            # print('video:', self._path_to_videos[index])
            video_container = None
            try:
                video_container = get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.debug(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.debug(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            if self.mode == "train": 
                frames, indexes = decode(
                    container=video_container, 
                    sampling_rate=sampling_rate, 
                    num_frames=self.cfg.DATA.NUM_FRAMES,
                    clip_idx=temporal_sample_index,
                    num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,
                    temporal_aug=self.mode == "train" and not self.cfg.DATA.NO_RGB_AUG,
                    rand_fr=self.cfg.DATA.RAND_FR #True
                )
            elif self.mode == "val":
                frames, indexes = decode(
                    container=video_container, 
                    sampling_rate=sampling_rate, 
                    num_frames=self.cfg.DATA.NUM_FRAMES,
                    clip_idx=temporal_sample_index,
                    num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,
                    # temporal_aug=self.mode == "train" and not self.cfg.DATA.NO_RGB_AUG,
                    temporal_aug=self.mode == "val" and not self.cfg.DATA.NO_RGB_AUG,
                    rand_fr=self.cfg.DATA.RAND_FR #True
                )
            
            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                warnings.warn(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[index]

            # if self.mode in ["test", "val"] or self.cfg.DATA.NO_RGB_AUG:
            if self.mode in ["test"] or self.cfg.DATA.NO_RGB_AUG:
                # Perform color normalization.
                frames = tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )

                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)

                # Perform data augmentation.
                frames = spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )
                if not self.cfg.MODEL.ARCH in ['vit']:
                    frames = pack_pathway_output(self.cfg, frames)
                else:
                    # Perform temporal sampling from the fast pathway.
                    frames = torch.index_select(
                        frames,
                        1,
                        torch.linspace(
                            0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                        ).long(),
                    )

            else:
                # T H W C -> T C H W.
                frames = [rearrange(x, "t h w c -> t c h w") for x in frames]

                # Perform data augmentation.
                augmentation = VideoDataAugmentationDINO()
                frames = augmentation(frames, from_list=True, no_aug=self.cfg.DATA.NO_SPATIAL)

                # T C H W -> C T H W.
                frames = [rearrange(x, "t c h w -> c t h w") for x in frames]

                # Perform temporal sampling from the fast pathway.
                frames = [torch.index_select(
                    x,
                    1,
                    torch.linspace(
                        0, x.shape[1] - 1,
                        x.shape[1] if self.cfg.DATA.RAND_FR else self.cfg.DATA.NUM_FRAMES

                    ).long(),
                ) for x in frames]
            
                # padding g1 to have the same number of frames as g2
                channel_0, frame_numbers_0, height_0, width_0 = frames[0].shape
                channel_1, frame_numbers_1, height_1, width_1 = frames[1].shape

                frames_to_add = frame_numbers_1 - frame_numbers_0

                if frames_to_add > 0:
                    padding = torch.zeros(channel_0, frames_to_add, height_0, width_0)
                    frames[0] = torch.cat((frames[0], padding), dim=1)
                
                # make g1 index equal to g2 index ( 4 --> 8) by adding the last value at the end of indexes
                desired_length = indexes[1].size(0)
                indexes[0] = torch.cat((indexes[0], torch.tensor([indexes[0][-1]] * (desired_length - len(indexes[0])))), dim=0)

            return frames, indexes, label

        else:
            print('skipping vide:', self._path_to_videos[index])
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )
            

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)