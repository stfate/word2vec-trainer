import argparse
from functools import partial
import tokenizer
import text_corpus
import word2vec_trainer


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--build-model", action="store_true", default=False)
    parser.add_argument("-o", "--output-model-path", default="model/word2vec.gensim.model")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=10)

    parser.add_argument("--corpus-path", default="data/TextCorpus")

    parser.add_argument("--download-neologd", action="store_true", default=False)
    parser.add_argument("--dictionary-path", default="output/dic")

    parser.add_argument("--use-pretrained-model", action="store_true", default=False)
    parser.add_argument("--pretrained-model-path", default="model/pretrained.word2vec.gensim.model")

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    options = get_options()

    commands = [
        "download_neologd",
        "build_model"
    ]
    if not any(options[c] for c in commands):
        print("At least one of following options needs to be specified:",
              *["  --" + c.replace("_", "-") for c in commands], sep="\n")

    dic_path = options["dictionary_path"]
    if options["download_neologd"]:
        tokenizer.download_neologd(dic_path)

    output_model_path = options["output_model_path"]
    size = options["size"]
    window = options["window"]
    min_count = options["min_count"]
    corpus_path = options["corpus_path"]
    use_pretrained_model = options["use_pretrained_model"]
    pretrained_model_path = options["pretrained_model_path"]
    corpus = text_corpus.ArtistReviewCorpus()
    if options["build_model"]:
        iter_docs = partial(corpus.iter_docs, corpus_path)
        tagger = tokenizer.get_tagger(dic_path)
        word2vec_trainer.train_word2vec_model(output_model_path, iter_docs, tagger, size, window, min_count, use_pretrained_model, pretrained_model_path)
