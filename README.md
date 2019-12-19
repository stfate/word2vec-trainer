word2vec-trainer
================================

# Overview

gensim word2vecモデルの学習を行うツールキット．


# Requirements

word2vecはgensimの実装を使用．
他の依存パッケージは`requirements.txt`を参照．


# Setup

```bash
git submodule init
git submodule update
pip install -r requirements.txt
```


# Run

## Wikipedia

モデルのハイパーパラメータを指定する．

```bash
python src/train_wikipedia.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --wikipedia-dump-path=$WIKIPEDIA_DUMP_PATH --size=100 --window=8 --min-count=5
```

パラメータ指定方法の詳細は`train_wikipedia.sh`を参照されたい．

## General dataset

```bash
python src/train_text_dataset.py -o $OUTPUT_PATH --dictionary-path=$DIC_PATH --corpus-path=$CORPUS_PATH --size=100 --window=8 --min-count=5
```

`TextDatasetBase`を継承したデータセットクラスを作成することで，任意のテキストデータセットに対し実行することが可能．

```python
class TextDatasetBase(ABC):
    """
    a bass class for text dataset
    
    Attributes
    ----------
    """
    @abstractmethod
    def iter_docs(self):
        """
        iterator of documents
        
        Parameters
        ----------
        """
        yield None
```

パラメータ指定方法の詳細は`train_text_dataset.sh`を参照されたい．


# How to use model

```python
model_path = "model/word2vec.gensim.model"
model = Word2Vec.load(model_path)
```
