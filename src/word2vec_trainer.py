import functools
from pathlib import Path
import multiprocessing
import logging

from gensim.models.word2vec import Word2Vec


logging.basicConfig(level=logging.INFO)


def count_generator(iter):
    return sum(1 for _ in iter)


def train_word2vec_model(output_model_path, iter_docs, tokenizer, size=300, window=8, min_count=5, sg=1, epoch=5, use_pretrained_model=False, pretrained_model_path=None):
    """
    Parameters
    ----------
    model_path : string
        Path of Word2Vec model
    iter_tokens : iterator
        Iterator of documents, which are lists of words
    """
    logging.info("get tokens iterator")

    iter_tokens = tokenizer.get_tokens_iterator(iter_docs, normalize=False)
    n_obs = count_generator(iter_tokens())

    logging.info("build vocabulary")

    if use_pretrained_model:
        model = Word2Vec.load(pretrained_model_path)
        model.build_vocab(iter_tokens(), update=True)
    else:
        model = Word2Vec(
            size=size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=multiprocessing.cpu_count()
        )
        model.build_vocab(iter_tokens(), update=False)
    
    logging.info("train word2vec")

    model.train(iter_tokens(), total_examples=n_obs, epochs=epoch)
    model.init_sims(replace=True)

    logging.info("save model")

    p = Path(output_model_path)
    if not p.parent.exists():
        p.parent.mkdir()
    model.save(output_model_path)

    logging.info("done.")
