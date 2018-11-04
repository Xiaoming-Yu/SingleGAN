MODE='multimodal'
# dataset details
CLASS=$1  #edges2shoes edges2handbags facades night2day
LOAD_SIZE=143
FINE_SIZE=128
INPUT_NC=3
LATENT_C_NUM=8
IS_FLIP=1

case ${CLASS} in
'facades')
  NITER=200
  NITER_DECAY=200
  SAVE_EPOCH=25
  ;;
'edges2shoes')
  NITER=15
  NITER_DECAY=15
  LOAD_SIZE=128
  SAVE_EPOCH=5
  IS_FLIP=0
  ;;
'edges2handbags')
  NITER=10
  NITER_DECAY=10
  LOAD_SIZE=128
  SAVE_EPOCH=5
  IS_FLIP=0
  ;;
'night2day')
  NITER=25
  NITER_DECAY=25
  SAVE_EPOCH=10
  ;;
*)
  echo 'Unknown category'${CLASS}
  ;;
esac

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
  --save_epoch_freq ${SAVE_EPOCH} \
  --is_flip ${IS_FLIP} \
  --display_port 8097 \
  --batchSize 1 \
  --ngf 64 \
  --ndf 64 \
  --nef 64

  




  
