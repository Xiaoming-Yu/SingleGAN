MODE='base'
# dataset details
CLASS=$1  #apple2orange summer2winter_yosemite horse2zebra
LOAD_SIZE=143
FINE_SIZE=128
INPUT_NC=3
NITER=100
NITER_DECAY=100

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
  --display_port 8097\
  --batchSize 1 \
  --ngf 64 \
  --ndf 64 
  




  
