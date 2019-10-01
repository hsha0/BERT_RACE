#export PATH=/usr/local/cuda-9.0/bin:$PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/

current_time=$(date "+%Y_%m_%d-%H_%M_%S")
BERT_BASE_DIR='BERT_BASE/uncased_L-12_H-768_A-12'
BERT_GC='gs://bert_sh/BERT_BASE/uncased_L-12_H-768_A-12'
DATA_PATH='/content/RACE'

python3 run_race_multiout.py \
 --data_dir=$DATA_PATH \
 --do_lower_case=True \
 --output_dir=gs://bert_sh/RACE_$current_time \
 --do_train=True \
 --do_eval=True \
 --task_name=middle \
 --vocab_file=$BERT_GC/vocab.txt \
 --bert_config_file=$BERT_GC/bert_config.json \
 --init_checkpoint=$BERT_GC/bert_model.ckpt \
 --max_seq_length=384 \
 --train_batch_size=32 \
 --eval_batch_size=8 \
 --learning_rate=5e-5 \
 --num_train_epochs=3.0 \
 --use_tpu=True \
 --tpu_name='grpc://10.109.36.98:8470'
