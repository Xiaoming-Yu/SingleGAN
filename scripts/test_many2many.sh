MODE='many2many'
# dataset details
CLASS=$1 #night_day_summer_winter
TIME_DIR=$2    # e.g. 2018_10_15_10_48_56
EPOCH='latest'
HOW_MANY=50
LOAD_SIZE=128
FINE_SIZE=128
INPUT_NC=3
DOMAIN_NUM=4


# training
GPU_ID=0
NAME=${MODE}_${CLASS}
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --checkpoints_dir ./checkpoints \
  --time_dir ${TIME_DIR} \
  --name ${NAME} \
  --mode ${MODE} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --d_num ${DOMAIN_NUM} \
  --how_many ${HOW_MANY} \
  --which_epoch ${EPOCH} \
  --no_flip \
  --batchSize 1 \
  --ngf 64 

  
