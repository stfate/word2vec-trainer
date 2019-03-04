
from abc import ABC, abstractmethod
import bz2
from pathlib import Path
import json
from glob import glob
import os
import re
import subprocess
import sys
import pylufia.nlp.textio as textio


class TextCorpusBase(ABC):
    @abstractmethod
    def iter_docs(self):
        yield None

class ArtistReviewCorpus(TextCorpusBase):
    def __init__(self):
        self.root_path = None

    def iter_docs(self, corpus_path):
        self.root_path = Path(corpus_path)
        corpus_flist = self.root_path.glob("contents/*/*.html.txt")
        meta = json.load(open(self.root_path / "metadata.json", "r"))
        for fn in corpus_flist:
            fid = fn.parent.stem
            title = meta[fid]
            reader = textio.TextReader(fn)
            lines = reader.read()
            yield {"title": title, "body": lines}

class WikipediaCorpus(TextCorpusBase):
    def __init__(self):
        self.__src_dir = os.path.dirname( os.path.abspath(__file__) )
        self.__wikiextractor_path = os.path.normpath("wikiextractor/WikiExtractor.py")
        self.__wikiextractor_cmd = os.path.join(self.__src_dir, "..", self.__wikiextractor_path)
        self.__wikiextractor_cmd = os.path.normpath(self.__wikiextractor_cmd)

    def iter_docs(self, file_path, dir_path):
        """
        Parameters
        ----------
        file_path : string
            File path of wikipedia dump file
        dir_path : string
            Directory path where extracted text files are put
        """

        extracted_file_path_pattern = os.path.join(dir_path, "*", "*.bz2")
        extracted_file_paths = glob(extracted_file_path_pattern)
        if len(extracted_file_paths) == 0:
            cmd = [self.__wikiextractor_cmd, "--compress", "--quiet", "-o", dir_path, file_path]
            subprocess.call(cmd, stdout=sys.stdout)
            extracted_file_paths = glob(extracted_file_path_pattern)

        re_doc_begin = re.compile(r"^<doc(\s.*)?>$")
        re_doc_end = re.compile(r"^</doc>$")
        re_nonstandard_namespace = re.compile(r".*:.*")
        for fpath in extracted_file_paths:
            with bz2.BZ2File(fpath) as f:
                is_inside_doc = False
                does_skip_this_doc = False
                lines = []
                for line in f:
                    line = line.decode("utf-8").rstrip()
                    if not is_inside_doc:
                        m = re_doc_begin.match(line)
                        if m is not None:
                            is_inside_doc = True
                            title = f.__next__().decode("utf-8").rstrip()
                            if re_nonstandard_namespace.match(title):
                                does_skip_this_doc = True
                            continue
                    else:
                        m = re_doc_end.match(line)
                        if m is not None:
                            if not does_skip_this_doc:
                                yield {"title": title, "body": "\n".join(lines)}
                            is_inside_doc = False
                            does_skip_this_doc = False
                            lines = []
                            continue
                        if len(line) > 0:
                            lines.append(line)

    def download_dump(self, file_path, url):
        cmd = ["curl", "-o", file_path, url]
        subprocess.call(cmd, stdout=sys.stdout)


# __src_dir = os.path.dirname( os.path.abspath(__file__) )
# __wikiextractor_path = os.path.normpath("wikiextractor/WikiExtractor.py")
# __wikiextractor_cmd = os.path.join(__src_dir, "..", __wikiextractor_path)
# __wikiextractor_cmd = os.path.normpath(__wikiextractor_cmd)

# def download_dump(file_path, url):
#     cmd = ["curl", "-o", file_path, url]
#     subprocess.call(cmd, stdout=sys.stdout)

# def iter_docs(file_path, dir_path):
#     """
#     Parameters
#     ----------
#     file_path : string
#         File path of wikipedia dump file
#     dir_path : string
#         Directory path where extracted text files are put
#     """

#     extracted_file_path_pattern = os.path.join(dir_path, "*", "*.bz2")
#     extracted_file_paths = glob(extracted_file_path_pattern)
#     if len(extracted_file_paths) == 0:
#         cmd = [__wikiextractor_cmd, "--compress", "--quiet", "-o", dir_path,
#                file_path]
#         subprocess.call(cmd, stdout=sys.stdout)
#         extracted_file_paths = glob(extracted_file_path_pattern)

#     re_doc_begin = re.compile(r"^<doc(\s.*)?>$")
#     re_doc_end = re.compile(r"^</doc>$")
#     re_nonstandard_namespace = re.compile(r".*:.*")
#     for fpath in extracted_file_paths:
#         with bz2.BZ2File(fpath) as f:
#             is_inside_doc = False
#             does_skip_this_doc = False
#             lines = []
#             for line in f:
#                 line = line.decode("utf-8").rstrip()
#                 if not is_inside_doc:
#                     m = re_doc_begin.match(line)
#                     if m is not None:
#                         is_inside_doc = True
#                         title = f.__next__().decode("utf-8").rstrip()
#                         if re_nonstandard_namespace.match(title):
#                             does_skip_this_doc = True
#                         continue
#                 else:
#                     m = re_doc_end.match(line)
#                     if m is not None:
#                         if not does_skip_this_doc:
#                             yield {"title": title, "body": "\n".join(lines)}
#                         is_inside_doc = False
#                         does_skip_this_doc = False
#                         lines = []
#                         continue
#                     if len(line) > 0:
#                         lines.append(line)
