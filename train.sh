#!/bin/bash

OUTPUT_PATH=model/wikipedia-ja-w2v-model/word2vec.gensim.model
DIC_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
WIKIPEDIA_DUMP_PATH=data/jawiki-latest-pages-articles.xml.bz2
SIZE=100
WINDOW=8
MIN_COUNT=5

# download wikipedia dump
# ./train --download-wikipedia-dump --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH

# download mecab-ipadic-neologd
# ./train --download-neologd --dictionary-path=$DIC_PATH

# train with wikipedia
./train --build-with-wikipedia -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH --size=$SIZE --window=$WINDOW --min-count=$MIN_COUNT
