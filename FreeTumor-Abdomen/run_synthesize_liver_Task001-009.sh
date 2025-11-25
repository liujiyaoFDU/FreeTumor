#!/bin/bash
conda activate freetumor
# ======TODO========
# TODO：测试集图像 / 标签
DATA_ROOT="/inspire/hdd/global_user/hejunjun-24017/jinye/data/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task002_lymph_node_CT"
TEST_DATA_PATH="${DATA_ROOT}/masked_imagesTs"
TEST_LABEL_PATH="${DATA_ROOT}/fixed_organ_labelsTs"
# TODO：合成图像 / 标签保存位置
TARGET_ROOT="/inspire/hdd/global_user/hejunjun-24017/jinye/data/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task002_lymph_node_CT"
SAVE_IMG_PATH="${TARGET_ROOT}/imagesTs"
SAVE_LAB_PATH="${TARGET_ROOT}/labelsTs"

bash run_synthesize_liver_utils.sh