TPU_NAME='grpc://10.23.23.154:8470'
BERT_GC='gs://bert_sh'
INIT_CKPT='gs://electra/electra_pretrain/bert_small_seq128_lr5e-4/model.ckpt-237000'
SEED=12345
TASK_INDEX=3

TASKS=(MRPC CoLA MNLI SST-2 QQP QNLI WNLI RTE STS-B)
LRS=(2e-5 1e-5 3e-5 1e-5 5e-5 1e-5 2e-5 3e-5 2e-5)
BZS=(32 16 128 32 128 32 16 32 16)
EPOCHS=(3 3 3 3 3 3 3 10 10)

TASK=${TASKS[${TASK_INDEX}]}
LR=${LRS[${TASK_INDEX}]}
BZ=${BZS[${TASK_INDEX}]}
EPOCH=${EPOCHS[${TASK_INDEX}]}

python3 run_classifier.py \
--bert_config_file=$BERT_GC/small_config.json \
--task_name=$TASK \
--data_dir=gs://electra/glue/glue_data/$TASK \
--output_dir=gs://electra/glue/glue_results/bert_small/${TASK}_${SEED} \
--init_checkpoint=$INIT_CKPT \
--vocab_file=vocab.txt \
--do_train=True \
--do_eval=True \
--train_batch_size=${BZ} \
--learning_rate=${LR} \
--max_seq_length=128 \
--num_train_epochs=${EPOCH} \
--seed=${SEED} \
--use_tpu=True \
--tpu_name=$TPU_NAME