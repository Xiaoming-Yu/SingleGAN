MODE='multimodal'
# dataset details
CLASS=$1  #edges2shoes edges2handbags
LOAD_SIZE=128
FINE_SIZE=128
INPUT_NC=3
LATENT_C_NUM=8
NITER=30
NITER_DECAY=30

# training
GPU_ID=0
DISPLAY_ID=$((GPU_ID*7+3))
NAME=${MODE}_${CLASS}
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --dataroot  ./datasets/${CLASS} \
  --checkpoints_dir ./checkpoints \
  --display_id ${DISPLAY_ID} \
  --name ${NAME} \
  --mode ${MODE} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --c_num ${LATENT_C_NUM} \
  --no_flip \
  --display_port 8097 \
  --batchSize 1 \
  --ngf 64 \
  --ndf 64 \
  --nef 64

  




  
