wikipedia-word2vec-model-builder
================================

# Overview

日本語Wikipediaダンプデータからgensim word2vecモデルの学習&word embeddingの計算を行うツールキット．

# Requirements

- cURL
- MeCab == 0.996
- Python >= 3.4

# Setup

```bash
git submodule init
git submodule update
pip install -r requirements.txt
```

# Run

## Training

WikipediaダンプファイルのDL & NEologd辞書ファイルDL & モデル学習 を行う．

```bash
./train --download-wikipedia-dump --download-neologd --build-gensim-model
```

モデルのハイパーパラメータを指定する．

```bash
./train --build-gensim-model --size=100 --window=8 --min-count=5
```

## Embedding extraction

```bash
./featext --model-path=path/to/model --dictionary-path=path/to/dictionary --output-directory=path/to/output
```

# How to use model

```python
model_path = "model/word2vec.gensim.model"
model = Word2Vec.load(model_path)
```
