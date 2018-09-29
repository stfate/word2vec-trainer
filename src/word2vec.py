# -*- coding: utf-8 -*-

import scipy as sp
from gensim.models.word2vec import Word2Vec
import functools
import os
import progressbar
import uuid
import json
import multiprocessing
import tokenizer


def count_generator(iter):
    return sum(1 for _ in iter)

def get_tokens_iterator(tagger, iter_docs):
    tokenize = functools.partial(tokenizer.tokenize, tagger=tagger)

    def iter_tokens():
        for doc in iter_docs():
            yield tokenize(doc["body"])

    return iter_tokens

def get_wordvector(model, word):
    if word in model.wv.vocab:
        return model.wv[word]
    else:
        return sp.zeros(model.vector_size, dtype=sp.float32)

def get_document_wordvector(model, words):
    n_words = len(words)
    doc_wv = sp.zeros( (n_words, model.vector_size), dtype=sp.float32 )
    for w,word in enumerate(words):
        doc_wv[w] = get_wordvector(model, word)

    return doc_wv

def train_gensim_word2vec_model(model_path, iter_docs, tagger, size, window, min_count, use_pretrained_model):
    """
    Parameters
    ----------
    model_path : string
        Path of Word2Vec model
    iter_tokens : iterator
        Iterator of documents, which are lists of words
    """
    iter_tokens = get_tokens_iterator(tagger, iter_docs)

    if use_pretrained_model:
        model = Word2Vec.load(model_path)
        update = True
    else:
        model = Word2Vec(
            size=size,
            sg=1,
            window=window,
            min_count=min_count,
            workers=multiprocessing.cpu_count()
        )
        update = False

    n_obs = count_generator( iter_tokens() )
    model.build_vocab( iter_tokens(), update=update )
    model.train(iter_tokens(), total_examples=n_obs, epochs=model.iter)
    model.init_sims(replace=True)
    model.save(model_path)

def extract_gensim_word2vec_features(model_path, iter_docs, tagger, output_dir):
    model = Word2Vec.load(model_path)
    metadata = {}
    bar = progressbar.ProgressBar()
    for doc in bar( iter_docs() ):
        doc_title = doc["title"]
        doc_text = doc["body"]
        doc_uuid = str( uuid.uuid3(uuid.NAMESPACE_URL, doc_title) )
        tokens = tokenizer.tokenize(doc_text, tagger)
        wv = get_document_wordvector(model, tokens)
        prefix = doc_uuid[:2]
        output_subdir = os.path.join(output_dir, prefix)
        if not os.path.exists(output_subdir):
            os.mkdir(output_subdir)
        output_fn = os.path.join( output_subdir, "{}.npy".format(doc_uuid) )
        sp.save(output_fn, wv)
        metadata[doc_uuid] = doc_title

    json.dump( metadata, open( "{}/metadata.json".format(output_dir), "w" ), ensure_ascii=False, indent=2 )
