# -*- coding: utf-8 -*-

"""
@package
@brief
@author stfate
"""

import glob
import pylufia.nlp.textio as textio


def iter_docs(corpus_path):
    corpus_flist = [fn for fn in glob.glob( "{}/contents/*".format(corpus_path) )]
    for fn in corpus_flist:
        reader = textio.TextReader(fn)
        lines = reader.read()
        yield lines
        