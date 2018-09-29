#!/bin/bash

./featext --wikipedia-dump-path=data/jawiki-latest-pages-articles.xml.bz2 --model-path=model/wikipedia-ja-w2v-model/word2vec.gensim.model --dictionary-path=/usr/lib/mecab/dic/mecab-ipadic-neologd --output-directory=feature/wikipedia-w2v
# ./build --wikipedia-dump-path=data/jawiki-latest-pages-articles.xml.bz2 --model-path=model/wikipedia-ja-w2v-model/word2vec.gensim.model --dictionary-path=/home/dsasai/local/lib/mecab/dic/mecab-ipadic-neologd --output-directory=feature/wikipedia-w2v
