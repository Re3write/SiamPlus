ROOT=/home/sk49/workspace/cy/Siamplus
export PYTHONPATH=$ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3

mkdir -p logs

python3 -u $ROOT/tools/train_siammask.py \
    --config=config.json -b 12 \
    -j 20\
    --epochs 20 \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

#bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4#
