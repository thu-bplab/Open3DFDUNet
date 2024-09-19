RESULTS_DIR='../results/xxx'
CATEGORY_NAME='xxxx'

python -u ../train.py \
    --cate $CATEGORY_NAME \
    --results_dir $RESULTS_DIR \
    --epochs 500 \
    --decay_epochs 400 450 \
    --decay_factors 0.5 0.1 \
    --eval_epoch 5 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-4
