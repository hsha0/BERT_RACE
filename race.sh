export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
BERT_BASE_DIR='BERT_BASE/uncased_L-12_H-768_A-12'

python3 run_race.py \
 --data_dir=RACE/train/middle \
 --do_lower_case=True \
 --output_dir=RACE_output \
 --do_train=True \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
 --max_seq_length=512 \
 --train_batch_size=32 \
 --learning_rate=2e-5 \
 --num_train_epochs=3.0 \
