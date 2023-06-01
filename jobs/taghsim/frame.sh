#!/bin/bash
#SBATCH --mail-user=sahar.dastani4776@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=brats
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-00:00
#SBATCH --account=rrg-ebrahimi

source ~/benchmark/bin/activate

cd /home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/severe_benchmark/action_detection_multi_label_classification/scripts/prepare-ava

echo "-----<d>-----"

IN_DATA_DIR="/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/taghsim/g"
OUT_DATA_DIR="/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/taghsim/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done
