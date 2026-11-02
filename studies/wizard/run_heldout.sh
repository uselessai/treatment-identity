#!/usr/bin/env bash
# The wizard on eight projects held out from its development: none was used to
# write or change it, and its code was not changed while they were run (its
# SHA-256 digests are in heldout_frozen.sha256). Each project runs the first
# training command its README gives (EDSR: the first of src/demo.sh, with the
# --dir_data its README says to set, since the default lies outside the
# project), in one process and without the distributed launcher, since the
# run has no GPU.
# Answers a person gives come from oracle/; nothing else is prepared but the
# environment.
cd "$(dirname "$0")/../.."
H=/home/DATA/laura/02ImproveData/ddc_wizard_heldout
C=/home/laura/miniconda3/envs
ENVS=/home/DATA/laura/02ImproveData/ddc_wizard_envs
P=$C/ddc_ext/bin/python
# No terminal for the wizard: its questions go to <key>.questions.json and the
# answers a person gives to oracle/<key>.json, as in an unattended run. (Under
# timeout a terminal question would stop the process without being shown.)
t() { local k=$1 d=$2; shift 2; timeout 3600 "${PY_FOR:-$P}" studies/wizard/trial.py "$k" "$d" "$@" </dev/null 2>&1 | tail -1; }

t nafnet $H/NAFNet -- python basicsr/train.py -opt options/train/REDS/NAFNet-width64.yml
t realesrgan $H/Real-ESRGAN -- python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml
t codeformer $H/CodeFormer -- python basicsr/train.py -opt options/VQGAN_512_ds32_nearest_stage1.yml
t retinexformer $H/Retinexformer -- python basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml
t hat $H/HAT -- python hat/train.py -opt options/train/train_HAT_SRx2_from_scratch.yml
t edsr $H/EDSR-PyTorch/src -- python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset \
    --dir_data ../dataset
# EDSR decodes every image when it is built and then reads only its own cache
# (--ext sep); --ext img, one of its options, reads the images for each item.
t edsr_img $H/EDSR-PyTorch/src -- python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset \
    --dir_data ../dataset --ext img
# Two projects without a configuration file: only command-line options, with
# their defaults in the code. SRCNN reads its training data from an HDF5 file.
t mimounet $H/MIMO-UNet -- python main.py --model_name MIMO-UNet --mode train --data_dir dataset/GOPRO
t srcnn $H/SRCNN-pytorch -- python train.py --train-file data/91-image_x3.h5 --eval-file data/Set5_x3.h5 \
    --outputs-dir outputs --scale 3 --lr 1e-4 --batch-size 16 --num-epochs 400 --num-workers 8 --seed 123
