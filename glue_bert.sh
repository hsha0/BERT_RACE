TPU_NAME='grpc://10.8.246.2:8470'
BERT_GC='gs://bert_sh'
INIT_CKPT=gs://bert_sh/bert_pretrain/bert_5e-4/model.ckpt-200000
TASK=MRPC

python3 run_classifier.py \
--bert_config_file=$BERT_GC/small_config.json \
--task_name=$TASK \
--data_dir=gs://electra/glue/glue_data/$TASK \
--output_dir=$BERT_GC/glue_results/bert_5e-4_200K/$TASK \
--init_checkpoint= $INIT_CKPT \
--vocab_file=vocab.txt \
--do_train=True \
--do_eval=True \
--train_batch_size=32 \
--learning_rate=3e-4 \
--max_seq_length=128 \
--num_train_epochs=3.0 \
--seed=12345 \
--use_tpu=True \
--tpu_name=$TPU_NAME