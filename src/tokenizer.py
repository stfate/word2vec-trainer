import os
import subprocess
import sys
import tempfile

import MeCab
import re


__neologd_repo_name = "mecab-ipadic-neologd"
__neologd_repo_url = "https://github.com/neologd/mecab-ipadic-neologd.git"

ptn_number = re.compile(r"([0-9]|[０-９])+")


def download_neologd(dic_path):
    dic_path = os.path.abspath(dic_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.call(["git", "clone", "--depth", "1", __neologd_repo_url],
                        stdout=sys.stdout, cwd=temp_dir)
        neologd_dir_path = os.path.join(temp_dir, __neologd_repo_name)
        subprocess.call(["./bin/install-mecab-ipadic-neologd", "-y", "-u",
                         "-p", dic_path],
                        stdout=sys.stdout, cwd=neologd_dir_path)


def get_tagger(dic_path):
    return MeCab.Tagger("-Ochasen -d {}".format(dic_path))

def concat_continuous_numbers(tokens):
    prev_token = ""
    tokens_out = tokens.copy()
    for i in range( 1, len(tokens_out) )[::-1]:
        cur_token = tokens_out[i]
        prev_token = tokens_out[i-1]
        if ptn_number.match(prev_token) and ptn_number.match(cur_token):
            tokens_out[i-1] = prev_token + cur_token
            tokens_out.pop(i)
            i = i - 1

    return tokens_out

def tokenize(text, tagger):
    tokens = []
    for line in tagger.parse(text).split('\n'):
        if line == "EOS":
            break
        surface = line.split('\t')[0]
        tokens.append(surface)

    tokens_mod = concat_continuous_numbers(tokens)
    return tokens_mod
