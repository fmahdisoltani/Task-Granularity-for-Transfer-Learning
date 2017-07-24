import yaml


class Parser(object):
    def __init__(self, path, video_preparation, label_preparation):
        self.annotations = self.open_annotations(path)
        self.video_preparation = video_preparation
        self.label_preparation = label_preparation

    def open_annotations(self, path):
        pass

    def prepare_videos(self):
        return self.video_preparation(self.annotations)

    def prepare_labels(self):
        return self.label_preparation(self.annotations)


class JsonParser
    def open_annotations(self, path):
        if path.endswith("gz"):
            with gzip.open(path, "rb") as f:
                json = pd.read_json(f.read().decode("ascii"))
        else:
            json = pd.read_json(path)
        return json





