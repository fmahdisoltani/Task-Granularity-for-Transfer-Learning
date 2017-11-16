import gzip
import os

import pandas as pd


class AnnotationParser(object):

    def __init__(self, annot_path, video_root,
                 file_path="file", caption_type="template"):
        self.video_root = video_root
        self.file_path = file_path
        self.caption_type = caption_type
        self.annotations = self.open_annotation(annot_path)

    @classmethod
    def open_annotation(cls, path):
        pass

    def get_video_paths(self):
        files = self.annotations[self.file_path]
        return [os.path.join(self.video_root, name.split("/")[0])
                for name in files]

    def get_video_ids(self):
        return [str(id) for id in self.annotations["id"]]

    def get_captions(self, caption_type=None):

        if caption_type is None:
            caption_type = self.caption_type
        return [p for p in self.annotations[caption_type]]

    def get_captions_from_tmp_and_lbl(self):
        return self.get_captions("template") + self.get_captions("label")

      
class JsonParser(AnnotationParser):

    @classmethod
    def open_annotation(cls, path):
        if path.endswith("gz"):
            with gzip.open(path, "rb") as f:
                json = pd.read_json(f.read().decode("utf-8"))
        else:
            json = pd.read_json(path)
        return json
