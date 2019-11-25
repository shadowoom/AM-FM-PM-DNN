#!/bin/bash

DATA_DIR=../../../data/twitter_data_concat

CUDA_VISIBLE_DEVICES=1 python -u inference.py \
--test_file=$DATA_DIR/test.txt \
--number_of_systems=21 \
--vocab_file=$DATA_DIR/vocab.txt \
--return_score=True \
--embedding_file=../../../data/twitter_embedding_w2v_d300_0.txt \
--output_dir=models/twitter_20191122_run \
--model_file=model_epoch_4.ckpt \
--maxlen_1=400 \
--maxlen_2=150 \
--hidden_size=300 \
--test_batch_size=8 \
--fix_embedding=True \
> eval_log.txt 2>&1 &

