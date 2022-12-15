# CL params
BUFFER_SIZE=500

# magnitude-based 1 shot retraining
ARCH="resnet" # 
DEPTH="18"
PRUNE_ARGS="--sp-retrain --sp-prune-before-retrain"
LOAD_CKPT="XXXXX.pth.tar"     # automatically train from scratch if the given checkpoint model is not found
INIT_LR="0.03"
EPOCHS="250"
WARMUP="8"

SPARSITY_TYPE="irregular"
DATASET="seq-cifar10"

GLOBAL_BATCH_SIZE="32"
MASK_UPDATE_DECAY_EPOCH="5-45"
SP_MASK_UPDATE_FREQ="5"

REMOVE_N=3000
RM_EPOCH=20

SAVE_FOLDER="checkpoints/resnet18/paper/gradient_effi/mutate_irr/${DATASET}/buffer_${BUFFER_SIZE}/"

PATH_TO_SPARCL=/home/zhan.zhe/SparCL # change to your own path
cd $PATH_TO_SPARCL

mkdir -p ${SAVE_FOLDER}

GPU_ID=0
SEED=888

GRADIENT=0.80
# ------- for 75% overall sparsity ----------
# ------- check retrain.py for more information ----------
LOWER_BOUND="0.75-0.76-0.75"
UPPER_BOUND="0.74-0.75-0.75"

CONFIG_FILE="./profiles/resnet18_cifar/irr/resnet18_0.75.yaml"
REMARK="irr_0.75_mut"
LOG_NAME="75_derpp_${GRADIENT}"
PKL_NAME="irr_0.75_mut_RM_${REMOVE_N}_${RM_EPOCH}"
# EVAL_CHECKPOINT="./checkpoints/resnet18/mutate_irr/seed914_irr_0.75_mut_resnet18_seq-cifar10_acc_57.110_fgt_43.500_sgd_lr0.1_cosine_sp0.000_task_4.pt"


CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u main_sparse_train_w_data_gradient_efficient.py \
        --arch ${ARCH} --depth ${DEPTH} --optmzr sgd --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler cosine --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --dataset ${DATASET} --seed ${SEED} --upper-bound ${UPPER_BOUND} --lower-bound ${LOWER_BOUND} --mask-update-decay-epoch ${MASK_UPDATE_DECAY_EPOCH} --sp-mask-update-freq ${SP_MASK_UPDATE_FREQ} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} \
        --log-filename=${SAVE_FOLDER}/seed_${SEED}_${LOG_NAME}.txt --buffer-size=$BUFFER_SIZE --replay_method derpp --buffer_weight 0.1 --buffer_weight_beta 0.5 \
        --use_cl_mask --gradient_efficient_mix --gradient_sparse=$GRADIENT --remove-n=$REMOVE_N --keep-lowest-n 0 --remove-data-epoch=$RM_EPOCH --output-dir ${SAVE_FOLDER} --output-name=${PKL_NAME}
        # --evaluate_mode --eval_checkpoint=${EVAL_CHECKPOINT}
