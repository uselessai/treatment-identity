#!/usr/bin/env bash
# The wizard on every loader of Studies 3 and 4: each project's training
# command, as its README or configuration gives it, run in that project's
# environment with DDC added. Answers a person gives come from oracle/.
#
# Environments: ddc_ext for most; the project's own environment where it has
# one (DART, DiffuEraser) or needs packages of its time (LaMa and E2FGVI in
# open-mmlab). ENVS holds packages those environments lack, put ahead of them.
# Commands run without a GPU; where a script requests one, the usual option
# for a CPU run is given.
cd "$(dirname "$0")/../.."
Z=/home/laura/02ImproveData
C=/home/laura/miniconda3/envs
ENVS=/home/DATA/laura/02ImproveData/ddc_wizard_envs
P=$C/ddc_ext/bin/python
# No terminal for the wizard: its questions go to <key>.questions.json and the
# answers a person gives to oracle/<key>.json, as in an unattended run. (Under
# timeout a terminal question would stop the process without being shown.)
t() { local k=$1 d=$2; shift 2; timeout 3600 "${PY_FOR:-$P}" studies/wizard/trial.py "$k" "$d" "$@" </dev/null 2>&1 | tail -1; }

t basicsr_reds $Z/zBasicSR -- python basicsr/train.py -opt options/train/BasicVSR/train_BasicVSR_REDS.yml
t restormer $Z/zRestormer -- python basicsr/train.py -opt Deraining/Options/Deraining_Restormer.yml
t gfpgan $Z/zGFPGAN -- python gfpgan/train.py -opt options/train_gfpgan_v1.yml
t ddcolor_lab $Z/zDDColor -- python basicsr/train.py -opt options/train/train_ddcolor.yml
EXTRA_PATH=$ENVS/envkair t kair_blindsr $Z/zKAIR -- python main_train_psnr.py --opt options/train_bsrgan_x4_psnr.json
t kair_video $Z/zKAIR -- python main_train_vrt.py --opt options/rvrt/001_train_rvrt_videosr_bi_reds_30frames.json
t vrt $Z/zKAIR -- python main_train_vrt.py --opt options/vrt/001_train_vrt_videosr_bi_reds_6frames.json
t zerodce $Z/zZero-DCE/Zero-DCE_code -- python lowlight_train.py --lowlight_images_path data/train_data/
t mprnet $Z/zMPRNet/Denoising -- python train.py
t uformer $Z/zUformer -- python train/train_denoise.py --arch Uformer_B --batch_size 32 --train_ps 128 --train_dir ../datasets/denoising/sidd/train --val_dir ../datasets/denoising/sidd/val
t pix2pix_colorization $Z/zpix2pix -- python train.py --dataroot ./datasets/colorization --name color_pix2pix --model colorization
EXTRA_PATH=$ENVS/envmmagic t mmagic $Z/zmmagic -- python tools/train.py configs/basicvsr/basicvsr_2xb4_reds4.py
PY_FOR=$C/open-mmlab/bin/python EXTRA_PATH=$ENVS/envlama:$ENVS/env38 t lama $Z/zLaMa TORCH_HOME=$Z/zLaMa -- \
    python bin/train.py -cn lama-fourier-celeba location=celeba_example \
    location.data_root_dir=$Z/zLaMa/celeba-hq-dataset/ location.out_root_dir=$Z/zLaMa/experiments/ \
    location.tb_dir=$Z/zLaMa/tb_logs/ location.pretrained_models=$Z/zLaMa/ \
    trainer.kwargs.gpus=0 trainer.kwargs.accelerator=null trainer.kwargs.num_sanity_val_steps=0
PY_FOR=$C/open-mmlab/bin/python EXTRA_PATH=$ENVS/env38 t e2fgvi $Z/zE2FGVI -- python train.py -c configs/train_e2fgvi.json
t propainter $Z/zProPainter -- python train.py -c configs/train_propainter.json
PY_FOR=$C/diffueraser/bin/python EXTRA_PATH=$ENVS/env39 t diffueraser $Z/zDiffuEraser/repo -- \
    python train_DiffuEraser_stage1.py --base_model_name_or_path=../weights/stable-diffusion-v1-5 \
    --brushnet_model_name_or_path=../weights/diffuEraser/brushnet --vae_path=../weights/sd-vae-ft-mse \
    --train_data_dir=data/train "--csv_file=['data/train/dataset1/metadata.csv']" --resolution=512 \
    --nframes=10 --mixed_precision=no --train_batch_size=1 --dataloader_num_workers=2 \
    --learning_rate=1e-05 --resume_from_checkpoint=latest --validation_steps=2000 \
    --output_dir=diffuEraser-model-stage1 --logging_dir=logs-stage1
PY_FOR=$Z/zDART/repo/.venv/bin/python EXTRA_PATH=$ENVS/env312 t dart $Z/zDART/repo -- \
    python scripts/train.py +trainer.num_sanity_val_steps=0 '~logger'
# The four old-film systems of Study 4, in copies of the audited clones (the
# clones themselves stay clean); the noise bank of their degradation is linked
# into each copy, and contracts/<key>.yml adds the requirement no
# configuration states.
O=/home/DATA/laura/02ImproveData/ddc_wizard_projects
t rtn $O/Bringing-Old-Films-Back-to-Life -- python VP_code/main_gan.py --name RNN_Swin_4 \
    --model_name RNN_Swin_4 --epoch 20 --nodes 1 --gpus 1 --discriminator_name discriminator_v2 --which_gan hinge
t rrtn $O/RRTN-old-film-restoration -- python VP_code/main_gan.py --name rrtn --num_recursion 1 --epoch 30 --gpus 1
t mambaofr $O/MambaOFR -- python VP_code/main_gan.py --name RNN_Swin_4 --model_name RNN_Swin_4 \
    --epoch 20 --nodes 1 --gpus 1 --discriminator_name discriminator_v2 --which_gan hinge
# MgfrOFR: its README names scripts/train.py, absent at this commit; train.py, which
# is present, imports basicofr.archs, which the repository does not contain
t mgfrofr $O/zMgfrOFR -- python train.py -opt options/train/mgfrofr_a3_full.yml
