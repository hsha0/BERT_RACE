#export PATH=/usr/local/cuda-9.0/bin:$PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/

TPU_NAME='grpc://10.21.242.202:8470'

BERT_GC='gs://bert_sh/BERT_LARGE/wwm_cased_L-24_H-1024_A-16'
DATA_PATH='/content/POS'



declare -a SEEDS=(12345)

for run in $(seq 1 1)
do
    current_time=$(date "+%y%m%d-%H%M%S")
    SEED=${SEEDS[$run-1]}
    python3 run_pos.py \
    --seed=$SEED \
    --data_dir=$DATA_PATH \
    --do_lower_case=False \
    --output_dir=gs://bert_sh/POS/POS_${SEED}_$current_time \
    --do_train=True \
    --do_eval=True \
    --do_predict=True \
    --vocab_file=$BERT_GC/vocab.txt \
    --bert_config_file=$BERT_GC/bert_config.json \
    --init_checkpoint=$BERT_GC/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=64 \
    --eval_batch_size=8 \
    --learning_rate=5e-5 \
    --num_train_epochs=3.0 \
    --use_tpu=True \
    --tpu_name=$TPU_NAME
done
