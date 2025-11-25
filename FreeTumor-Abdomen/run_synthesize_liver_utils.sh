#!/bin/bash
# ======================
# 设置模型路径
TGAN_CHECKPOINT="/inspire/hdd/global_user/hejunjun-24017/jiyao/Project/202504_3D生成/20251015_tumor_synthesis/model/FreeTumor/baseline/TGAN.pt"
PRETRAINED_DIR="/inspire/hdd/global_user/hejunjun-24017/jiyao/Project/202504_3D生成/20251015_tumor_synthesis/model/FreeTumor/baseline/model_baseline_segmentor.pt"

# 创建保存目录
mkdir -p ${SAVE_IMG_PATH}
mkdir -p ${SAVE_LAB_PATH}

echo "======================================================================"
echo "FreeTumor Liver Tumor Synthesis Test"
echo "======================================================================"
echo "Test Data Path: ${TEST_DATA_PATH}"
echo "Test Label Path: ${TEST_LABEL_PATH}"
echo "TGAN Checkpoint: ${TGAN_CHECKPOINT}"
echo "Pretrained Dir: ${PRETRAINED_DIR}"
echo "Save Image Path: ${SAVE_IMG_PATH}"
echo "Save Label Path: ${SAVE_LAB_PATH}"
echo "======================================================================"

# 运行肝脏肿瘤合成测试
python Syn_data/Synthesize_liver.py \
    --test_data_path ${TEST_DATA_PATH} \
    --test_label_path ${TEST_LABEL_PATH} \
    --save_img_path ${SAVE_IMG_PATH} \
    --save_lab_path ${SAVE_LAB_PATH} \
    --TGAN_checkpoint ${TGAN_CHECKPOINT} \
    --pretrained_dir ${PRETRAINED_DIR} \
    --out_channels 7 \
    --feature_size 48 \
    --roi_x 96 \
    --roi_y 96 \
    --roi_z 96 

echo "======================================================================"
echo "Test completed! Check results in:"
echo "  Images: ${SAVE_IMG_PATH}"
echo "  Labels: ${SAVE_LAB_PATH}"
echo "======================================================================"
