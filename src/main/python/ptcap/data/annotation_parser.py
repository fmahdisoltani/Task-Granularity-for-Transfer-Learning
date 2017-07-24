import gzip
import os
import pandas as pd


class AnnotationParser(object):
    FILE_FIELD = "file"
    CAPTION_FIELD = "label"

    def __init__(self, path):
        self.annotations = self.open_annotation(path)


    @classmethod
    def open_annotation(cls, path):
        if path.endswith("gz"):
            with gzip.open(path, "rb") as f:
                json = pd.read_json(f.read().decode("ascii"))
        else:
            json = pd.read_json(path)
        return json

    def get_video_paths(self, annotations, root):
        files = list(annotations[self.FILE])
        return [os.path.join(root, str(name)) for name in files]