#!/bin/bash

# change this to your preferred download location
PRETRAINED_MODELS_PATH=./pretrained_models

# GLIP model
mkdir -p $PRETRAINED_MODELS_PATH/GLIP/checkpoints
mkdir -p $PRETRAINED_MODELS_PATH/GLIP/configs
wget -nc -P $PRETRAINED_MODELS_PATH/GLIP/checkpoints https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
wget -nc -P $PRETRAINED_MODELS_PATH/GLIP/configs https://raw.githubusercontent.com/microsoft/GLIP/main/configs/pretrain/glip_Swin_L.yaml

# X-VLM model
mkdir -p $PRETRAINED_MODELS_PATH/xvlm
gdown "https://drive.google.com/u/0/uc?id=1bv6_pZOsXW53EhlwU0ZgSk03uzFI61pN" -O $PRETRAINED_MODELS_PATH/xvlm/retrieval_mscoco_checkpoint_9.pth

# TCL model
mkdir -p $PRETRAINED_MODELS_PATH/TCL
# this did not work: gdown "https://drive.google.com/uc?id=1Cb1azBdcdbm0pRMFs-tupKxILTCXlB4O" -O $PRETRAINED_MODELS_PATH/TCL/TCL_4M.pth
# instead, I found this to work:
gdown --id 1eHinvFP7TnZYAL2Ft-M8rPott7mpVN2R -O TCL.zip
# then unzip it to the TCL folder
unzip TCL.zip -d $PRETRAINED_MODELS_PATH/TCL
# we need the TCL_4M.pth file only, so we can delete the rest
rm TCL.zip
#https://drive.google.com/file/d/1eHinvFP7TnZYAL2Ft-M8rPott7mpVN2R/view 
# InSPyReNet model
mkdir -p $PRETRAINED_MODELS_PATH/saliency_inspyrenet_plus_ultra
gdown "https://drive.google.com/uc?id=13oBl5MTVcWER3YU4fSxW3ATlVfueFQPY" -O $PRETRAINED_MODELS_PATH/saliency_inspyrenet_plus_ultra/latest.pth
