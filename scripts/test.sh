MODEL_PATH='../results/xxx/model_epoch_xxx.pth'
RESULTS_DIR='../results/xxx/xxx'
CATEGORY_NAME='xxxx'

python ../test.py \
   --cate $CATEGORY_NAME \
   --model_path $MODEL_PATH \
   --batch_size 16 \
   --results_dir $RESULTS_DIR 
