#!/bin/bash

# DATASET_PATH=../../dataset/ArtistReviewCorpus_20180608
# LANG=ja
DATASET_PATH=../../dataset/MARD
LANG=en
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
# OUTPUT_PATH=model/corpus-w2v-model/word2vec.gensim.model
OUTPUT_PATH=model/mard-w2v-model/word2vec.gensim.model
PRETRAINED_MODEL_PATH=model/wikipedia-ja-w2v-model/word2vec.gensim.model
SIZE=300
WINDOW=8
MIN_COUNT=1

# download mecab-ipadic-neologd
# python src/train_text_dataset.py --download-neologd --dictionary-path=$DIC_PATH

# python src/train_text_dataset.py --build-model -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --dataset-path=$DATASET_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --use-pretrained-model --pretrained-model-path=$PRETRAINED_MODEL_PATH
python src/train_text_dataset.py --build-model -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --dataset-path=$DATASET_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT
