#!/bin/bash

WIKIPEDIA_DUMP_PATH=../../dataset/Wikipedia/jawiki-latest-pages-articles.xml.bz2
LANG=ja
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
OUTPUT_PATH=model/jawiki-w2v-model/word2vec.gensim.model
SIZE=300
WINDOW=8
MIN_COUNT=5
SG=1
EPOCH=5

# download wikipedia dump
# python src/train_wikipedia.py --download-wikipedia-dump --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH

# download mecab-ipadic-neologd
# python src/train_wikipedia.py --download-neologd --dictionary-path=$DIC_PATH

# train word2vec model
python src/train_wikipedia.py --build-model -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH --lang=$LANG --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT --sg=$SG --epoch=$EPOCH
