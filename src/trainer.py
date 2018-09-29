# -*- coding: utf-8 -*-

import argparse
from functools import partial
import tempfile
import glob

import tokenizer
import text_corpus
import wikipedia
import word2vec

__jawiki_dump_file_name = "jawiki-latest-pages-articles.xml.bz2"
__jawiki_dump_url = "https://dumps.wikimedia.org/jawiki/latest/{}".format(__jawiki_dump_file_name)  # noqa


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--train-with-wikipedia", action="store_true", default=False )
    parser.add_argument( "--train-with-corpus", action="store_true", default=False)
    parser.add_argument( "-o", "--output-model-path", default="model/word2vec.gensim.model" )
    parser.add_argument( "--size", type=int, default=100 )
    parser.add_argument( "--window", type=int, default=8 )
    parser.add_argument( "--min-count", type=int, default=10 )

    parser.add_argument( "--download-wikipedia-dump", action="store_true", default=False )
    parser.add_argument( "--wikipedia-dump-path", default="data/{}".format(__jawiki_dump_file_name) )
    parser.add_argument( "--wikipedia-dump-url", default=__jawiki_dump_url )

    parser.add_argument( "--corpus-path", default="corpus")

    parser.add_argument( "--download-neologd", action="store_true", default=False )
    parser.add_argument( "--dictionary-path", default="output/dic" )
    parser.add_argument( "--use-pretrained-model", action="store_true", default=False )

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    options = get_options()

    commands = [
        "download_neologd",
        "download_wikipedia_dump",
        "train_with_wikipedia",
        "train_with_corpus"
    ]
    if not any(options[c] for c in commands):
        print("At least one of following options needs to be specified:",
              *["  --" + c.replace("_", "-") for c in commands], sep="\n")

    dic_path = options["dictionary_path"]
    if options["download_neologd"]:
        tokenizer.download_neologd(dic_path)

    wikipedia_dump_path = options["wikipedia_dump_path"]
    wikipedia_dump_url = options["wikipedia_dump_url"]
    if options["download_wikipedia_dump"]:
        wikipedia.download_dump(wikipedia_dump_path, wikipedia_dump_url)

    corpus_path = options["corpus_path"]

    output_model_path = options["output_model_path"]
    size = options["size"]
    window = options["window"]
    min_count = options["min_count"]
    use_pretrained_model = options['use_pretrained_model']
    
    if options["train_with_wikipedia"]:
        with tempfile.TemporaryDirectory() as temp_dir:
            iter_docs = partial(wikipedia.iter_docs, wikipedia_dump_path, temp_dir)
            tagger = tokenizer.get_tagger(dic_path)
            # iter_tokens = get_tokens_iterator(tokenizer.get_tagger(dic_path), iter_docs)
            word2vec.train_gensim_word2vec_model(output_model_path, iter_docs, tagger, size, window, min_count, use_pretrained_model)

    if options["train_with_corpus"]:
        iter_docs = partial(text_corpus.iter_docs, corpus_path)
        tagger = tokenizer.get_tagger(dic_path)
        # iter_tokens = get_tokens_iterator(tokenizer.get_tagger(dic_path), iter_docs)
        word2vec.train_gensim_word2vec_model(output_model_path, iter_docs, tagger, size, window, min_count, use_pretrained_model)
