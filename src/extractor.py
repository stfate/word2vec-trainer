#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@package
@brief
@author stfate
"""

import argparse
import functools
import tempfile
import tokenizer
import wikipedia
import text_corpus
import word2vec

__jawiki_dump_file_name = "jawiki-latest-pages-articles.xml.bz2"
__jawiki_dump_url = "https://dumps.wikimedia.org/jawiki/latest/{}".format(__jawiki_dump_file_name)  # noqa


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--extract-with-wikipedia", action="store_true", default=False)
    parser.add_argument( "--extract-with-corpus", action="store_true", default=False)
    parser.add_argument( "--wikipedia-dump-path", default="data/{}".format(__jawiki_dump_file_name) )
    parser.add_argument( "--model-path", default="model/word2vec.gensim.model" )
    parser.add_argument( "--dictionary-path", default="output/dic" )
    parser.add_argument( "--corpus-path", default="corpus" )
    parser.add_argument( "--output-directory", default="output" )

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    options = get_options()

    commands = [
        "extract_with_wikipedia",
        "extract_with_corpus"
    ]
    if not any(options[c] for c in commands):
        print("At least one of following options needs to be specified:",
              *["  --" + c.replace("_", "-") for c in commands], sep="\n")

    wikipedia_dump_path = options["wikipedia_dump_path"]
    w2v_model_path = options["model_path"]
    dic_path = options["dictionary_path"]
    corpus_path = options["corpus_path"]
    output_dir = options["output_directory"]

    if options["extract_with_wikipedia"]:
        with tempfile.TemporaryDirectory() as temp_dir:
            iter_docs = functools.partial(wikipedia.iter_docs, wikipedia_dump_path, temp_dir)
            tagger = tokenizer.get_tagger(dic_path)
            word2vec.extract_gensim_word2vec_features(w2v_model_path, iter_docs, tagger, output_dir)

    if options["extract_with_corpus"]:
        iter_docs = functools.partial(text_corpus.iter_docs, corpus_path)
        tagger = tokenizer.get_tagger(dic_path)
        word2vec.extract_gensim_word2vec_features(w2v_model_path, iter_docs, tagger, output_dir)
