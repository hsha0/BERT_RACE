TPU_NAME='grpc://10.43.245.90:8470'
BERT_GC='gs://bert_sh'
LR=5e-4


MAX_SEQ_L=128

python3 run_pretraining.py \
--input_file=gs://electra/data_128_sent_CLS/*.tfrecord \
--output_dir=$BERT_GC/bert_pretrain/bert_small_seq${MAX_SEQ_L}_${LR} \
--bert_config_file=$BERT_GC/small_config.json \
--vocab_file=vocab.txt \
--do_train=True \
--learning_rate=${LR} \
--train_batch_size=128 \
--max_seq_length=${MAX_SEQ_L} \
--num_train_steps=1 \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=6250 \
--iterations_per_loop=6250 \
--use_tpu=true \
--tpu_name=$TPU_NAME
