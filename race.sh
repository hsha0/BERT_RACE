#export PATH=/usr/local/cuda-9.0/bin:$PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/


current_time=$(date "+%y%m%d-%H%M%S")

TPU_NAME='10.26.122.122:8470'
TASK_NAME='middle'

BERT_BASE_DIR='BERT_BASE/uncased_L-12_H-768_A-12'
BERT_GC='gs://bert_sh/BERT_BASE/uncased_L-12_H-768_A-12'
DATA_PATH='/content/RACE'

declare -a SEEDS=(699203 332037 5591 99716 676765 785600 65274)

for run in $(seq 1 7)
do
    SEED=${SEEDS[$run-1]}
    python3 run_race.py \
    --seed=$SEED \
    --data_dir=$DATA_PATH \
    --do_lower_case=True \
    --output_dir=gs://bert_sh/predict_10seed/RACE_${TASK_NAME}_${SEED}_$current_time \
    --do_train=True \
    --do_eval=True \
    --do_predict=True \
    --task_name=$TASK_NAME \
    --vocab_file=$BERT_GC/vocab.txt \
    --bert_config_file=$BERT_GC/bert_config.json \
    --init_checkpoint=$BERT_GC/bert_model.ckpt \
    --max_seq_length=384 \
    --train_batch_size=64 \
    --eval_batch_size=8 \
    --learning_rate=5e-5 \
    --num_train_epochs=3.0 \
    --use_tpu=True \
    --tpu_name=$TPU_NAME
done
