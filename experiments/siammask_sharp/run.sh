if [ -z "$1" ]
  then
    echo "Need input base model!"
    echo "Usage: bash `basename "$0"` \$BASE_MODEL"
    exit
fi


ROOT=/home/sk49/workspace/cy/Siamplus
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

base=$1

python3 -u $ROOT/tools/train_siammask_refine.py \
    --config=config.json -b 64 \
    -j 20 --pretrained $base \
    --epochs 20 \
    2>&1 | tee logs/train.log
