#!/bin/bash
# conda activate freetumor
# Seed = 2025

# 定义任务列表
TASKS=(
    "Task001_nasopharynx_tumor_CT"
    "Task002_lymph_node_CT"
    "Task003_lung_tumor_CT"
    "Task004_liver_tumor_CT"
    "Task005_kidney_tumor_CT"
    "Task006_pancreatic_tumor_CT"
    "Task007_colon_cancer_primaries_CT"
    "Task009_bladder_tumor_T2W_MRI"
)

# 设置模型路径
TGAN_CHECKPOINT="/inspire/hdd/global_user/hejunjun-24017/jiyao/Project/202504_3D生成/20251015_tumor_synthesis/model/FreeTumor/baseline/TGAN.pt"
PRETRAINED_DIR="/inspire/hdd/global_user/hejunjun-24017/jiyao/Project/202504_3D生成/20251015_tumor_synthesis/model/FreeTumor/baseline/model_baseline_segmentor.pt"

# 设置seed
SEED=2025

echo "======================================================================"
echo "FreeTumor Synthesis with SEED=${SEED}"
echo "======================================================================"

# 循环处理每个任务
for TASK in "${TASKS[@]}"; do
    echo "======================================================================"
    echo "Processing ${TASK} with seed=${SEED}"
    echo "======================================================================"

    # 设置数据路径
    DATA_ROOT="/inspire/hdd/global_user/hejunjun-24017/jinye/data/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/${TASK}"
    TEST_DATA_PATH="${DATA_ROOT}/masked_imagesTs"
    TEST_LABEL_PATH="${DATA_ROOT}/fixed_organ_labelsTs"

    # 设置保存路径（格式：{seed}_imagesTs），将 Task00X 转换为 Task20X
    TARGET_TASK="${TASK/Task00/Task20}"
    TARGET_ROOT="/inspire/hdd/global_user/hejunjun-24017/jinye/data/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/${TARGET_TASK}"
    SAVE_IMG_PATH="${TARGET_ROOT}/${SEED}_imagesTs"
    SAVE_LAB_PATH="${TARGET_ROOT}/${SEED}_labelsTs"

    # 创建保存目录
    mkdir -p ${SAVE_IMG_PATH}
    mkdir -p ${SAVE_LAB_PATH}

    echo "Test Data Path: ${TEST_DATA_PATH}"
    echo "Test Label Path: ${TEST_LABEL_PATH}"
    echo "Save Image Path: ${SAVE_IMG_PATH}"
    echo "Save Label Path: ${SAVE_LAB_PATH}"
    echo "======================================================================"

    # 运行肿瘤合成
    python Syn_data/Synthesize_liver.py \
        --test_data_path ${TEST_DATA_PATH} \
        --test_label_path ${TEST_LABEL_PATH} \
        --save_img_path ${SAVE_IMG_PATH} \
        --save_lab_path ${SAVE_LAB_PATH} \
        --TGAN_checkpoint ${TGAN_CHECKPOINT} \
        --pretrained_dir ${PRETRAINED_DIR} \
        --seed ${SEED} \
        --out_channels 7 \
        --feature_size 48 \
        --roi_x 96 \
        --roi_y 96 \
        --roi_z 96

    echo "======================================================================"
    echo "${TASK} completed! Results saved in:"
    echo "  Images: ${SAVE_IMG_PATH}"
    echo "  Labels: ${SAVE_LAB_PATH}"
    echo "======================================================================"
    echo ""
done

echo "======================================================================"
echo "All tasks completed with SEED=${SEED}!"
echo "======================================================================"
