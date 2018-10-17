MODE='many2many'
# dataset details
CLASS=$1 #night_day_summer_winter
LOAD_SIZE=143
FINE_SIZE=128
INPUT_NC=3
DOMAIN_NUM=4
NITER=50
NITER_DECAY=50

# training
GPU_ID=0
DISPLAY_ID=$((GPU_ID*7+3))
NAME=${MODE}_${CLASS}
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --dataroot ./datasets/${CLASS} \
  --checkpoints_dir ./checkpoints \
  --display_id ${DISPLAY_ID} \
  --name ${NAME} \
  --mode ${MODE} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --d_num ${DOMAIN_NUM} \
  --display_port 8097 \
  --batchSize 1 \
  --ngf 64 \
  --ndf 64 
  




  
