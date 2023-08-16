# Copyright (c) OpenMMLab. All rights reserved.
import os
import subprocess
from functools import partial
from multiprocessing.pool import Pool

import hydra
import tqdm
from omegaconf import DictConfig

from src import configs_dir


def downscale_clip(inname, outname):
    inname = '"%s"' % inname
    outname = '"%s"' % outname
    command = f"ffmpeg -i {inname} -filter:v scale=\"trunc(oh*a/2)*2:256\" -q:v 1 -c:a copy {outname}"
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(err)
        return err.output

    return output


def downscale_clip_wrapper(folder_path, output_path, file_name):
    in_name = f"{folder_path}/{file_name}"
    out_name = f"{output_path}/{file_name}"

    log = downscale_clip(in_name, out_name)
    return file_name, log


@hydra.main(version_base=None, config_path=configs_dir(),
            config_name="config")
def resize_videos(cfg: DictConfig) -> None:
    root_path = os.path.expanduser(cfg['src_dir'])
    split = cfg['split']

    folder_path = f'{root_path}/{split}'
    output_path = f'{root_path}/{split}_256'
    os.makedirs(output_path, exist_ok=True)

    file_list = os.listdir(folder_path)
    completed_file_list = set(os.listdir(output_path))
    file_list = [x for x in file_list if x not in completed_file_list]

    print(f"Starting to downsample {len(file_list)} video files.")

    downscale_clip_wrapper_with_args = partial(downscale_clip_wrapper, folder_path, output_path)
    with Pool(processes=cfg['num_workers']) as p:
        list(tqdm.tqdm(p.imap(downscale_clip_wrapper_with_args, file_list), total=len(file_list)))


if __name__ == '__main__':
    resize_videos()
