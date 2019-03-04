word2vec-trainer
================================

# Overview

gensim word2vecモデルの学習を行うツールキット．

# Requirements

- cURL
- MeCab == 0.996
- Python >= 3.6

# Setup

```bash
git submodule init
git submodule update
pip install -r requirements.txt
```

# Run

## Wikipedia

WikipediaダンプファイルのDL & NEologd辞書ファイルDL & モデル学習 を行う．

```bash
python src/train_wikipedia.py --download-wikipedia-dump --download-neologd --build-model
```

モデルのハイパーパラメータを指定する．

```bash
python src/train_wikipedia.py --build-model --size=100 --window=8 --min-count=5
```

## General corpus

```bash
python src/train_text_corpus.py --build-model --corpus-path=CORPUS_PATH --size=100 --window=8 --min-count=5
```

# How to use model

```python
model_path = "model/word2vec.gensim.model"
model = Word2Vec.load(model_path)
```
