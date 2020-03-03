TPU_NAME='grpc://10.49.215.154:8470'
LR=5e-4
TRAIN_STEP=179000

MAX_SEQ_L=128

python3 run_pretraining.py \
--input_file=gs://electra/data_128_sent_CLS/*.tfrecord \
--output_dir=gs://electra/electra_pretrain/bert_small_seq${MAX_SEQ_L}_lr${LR}_lm_weight \
--bert_config_file=small_config.json \
--vocab_file=vocab.txt \
--do_train=True \
--learning_rate=${LR} \
--train_batch_size=1024 \
--max_seq_length=${MAX_SEQ_L} \
--num_train_steps=${TRAIN_STEP} \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=1000 \
--iterations_per_loop=1000 \
--use_tpu=true \
--tpu_name=$TPU_NAME
