#!/bin/bash
find . -type f -name "*.pyc" -delete

export GPUID=0
export NET="squeezeSeg"
export IMAGE_SET="train"
export LOG_DIR="./log_train_single/"
export STEPS=50

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/train.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-gpu                      gpu id"
  echo "-image_set                (train|val)"
  echo "-log_dir                  Where to save logs."
  echo "-steps                    Number of training steps."
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-gpu                      gpu id"
      echo "-image_set                (train|val)"
      echo "-log_dir                  Where to save logs."
      echo "-steps                    Number of training steps."
      exit 0
      ;;
    -gpu)
      export GPUID="$2"
      shift
      shift
      ;;
    -image_set)
      export IMAGE_SET="$2"
      shift
      shift
      ;;
    -log_dir)
      export LOG_DIR="$2"
      shift
      shift
      ;;
    -steps)
      export STEPS="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

logdir="$LOG_DIR/"
echo "$logdir"

python ./src/train.py \
  --dataset=KITTI \
  --pretrained_model_path='./data/SqueezeNet/squeezenet_v1.1.pkl' \
  --data_path=./data/gazebo_data \
  --image_set=$IMAGE_SET \
  --train_dir="$logdir/train" \
  --net=$NET \
  --max_steps=$STEPS \
  --summary_step=50 \
  --checkpoint_step=50 \
  --gpu=$GPUID
