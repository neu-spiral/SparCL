cd ../../..

# magnitude-based 1 shot retraining
ARCH=${1:-"resnet"} # 
DEPTH=${2:-"18"}
PRUNE_ARGS=${5:-"--sp-retrain --sp-prune-before-retrain"}
LOAD_CKPT=${6:-"XXXXX.pth.tar"}
INIT_LR=${8:-"0.1"}
EPOCHS=${11:-"250"}
WARMUP=${12:-"8"}

DATASET=${20:-"seq-cifar10"}

GLOBAL_BATCH_SIZE=${9:-"32"}
MASK_UPDATE_DECAY_EPOCH=${18:-"25-40"}
# MASK_UPDATE_DECAY_EPOCH=${18:-"90-120"}
SP_MASK_UPDATE_FREQ=${19:-"2"}

REMOVE_N=${20:-"2000"}
KEEP_LOWEST_N=${21:-"0"}

GPU_ID=${15:-"0"}
SEED=${10:-"914"}

RM_EPOCH=${23:-"20"}
GRADIENT=${25:-"0.80"}

# --------------------------------------------------------------------------
SPARSITY_TYPE=${3:-"irregular"}
SAVE_FOLDER=${7:-"checkpoints/resnet18/data_sparse/mutate_irr/${DATASET}/remove_${REMOVE_N}_${GRADIENT}_cl_mask/"}

mkdir -p ${SAVE_FOLDER}

LOWER_BOUND=${17:-"0.77-0.76-0.75"}
UPPER_BOUND=${16:-"0.75-0.75-0.75"}

CONFIG_FILE=${4:-"./profiles/resnet18_cifar/irr/resnet18_0.75.yaml"}

REMARK=${13:-"irr_0.75_mut_RM_${REMOVE_N}"}
LOG_NAME=${14:-"irr_0.75_mut_RM_${REMOVE_N}_${RM_EPOCH}"}
PKL_NAME=${23:-"irr_0.75_mut_RM_${REMOVE_N}_${RM_EPOCH}"}
BUFFER_SIZE=${24:-"500"}


CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u main_sparse_train_w_data_gradient_efficient.py \
    --arch ${ARCH} --depth ${DEPTH} --optmzr sgd --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler cosine --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --dataset ${DATASET} --seed ${SEED} --upper-bound ${UPPER_BOUND} --lower-bound ${LOWER_BOUND} --mask-update-decay-epoch ${MASK_UPDATE_DECAY_EPOCH} --sp-mask-update-freq ${SP_MASK_UPDATE_FREQ} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} \
    --log-filename=${SAVE_FOLDER}/seed_${SEED}_${LOG_NAME}.txt --output-dir ${SAVE_FOLDER} --output-name ${PKL_NAME} --remove-n ${REMOVE_N} --keep-lowest-n ${KEEP_LOWEST_N} \
    --remove-data-epoch ${RM_EPOCH} --buffer-size ${BUFFER_SIZE} --gradient_efficient_mix --gradient_sparse ${GRADIENT}