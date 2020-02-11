TPU_NAME='grpc://10.93.191.82:8470'
BERT_GC='gs://bert_sh'

python3 run_pretraining.py \
--input_file=gs://electra/data_128_CLS/*.tfrecord \
--output_dir=$BERT_GC/bert_pretrain/bert_5e-4 \
--bert_config_file=$BERT_GC/small_config.json \
--vocab_file=vocab.txt \
--do_train=True \
--learning_rate=5e-4 \
--train_batch_size=128 \
--max_seq_length=128 \
--num_train_steps=1000000 \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=12500 \
--iterations_per_loop=12500 \
--use_tpu=true \
--tpu_name=$TPU_NAME
