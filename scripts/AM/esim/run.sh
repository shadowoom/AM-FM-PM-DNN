#!/bin/bash

DATA_DIR=../../../data/twitter_data_concat

CUDA_VISIBLE_DEVICES=2 python -u main.py \
--train_file=$DATA_DIR/train.txt \
--valid_file=$DATA_DIR/valid.txt \
--test_file=$DATA_DIR/test.txt \
--vocab_file=$DATA_DIR/vocab.txt \
--output_dir=twitter_20191122_run \
--embedding_file=../../data/twitter_embedding_w2v_d300.txt \
--maxlen_1=400 \
--maxlen_2=150 \
--hidden_size=300 \
--train_batch_size=8 \
--valid_batch_size=8 \
--test_batch_size=8 \
--fix_embedding=True \
--patience=1 \
> log.txt 2>&1 &

