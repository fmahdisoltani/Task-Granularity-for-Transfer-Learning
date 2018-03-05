import gzip
import os

import pandas as pd


class AnnotationParser(object):

    def __init__(self, annot_path, video_root, file_path="file",
                 caption_type="template", object_list=None, single_object=False):
        self.video_root = video_root
        self.file_path = file_path
        self.caption_type = caption_type
        self.single_object = single_object
        self.annotations = self.open_annotation(annot_path)
        self.object_list = object_list

    @classmethod
    def open_annotation(cls, path):
        pass

    def get_video_paths(self):
        return NotImplementedError
    def get_video_ids(self):
        return NotImplementedError
    def get_captions(self):
        return NotImplementedError

    def get_captions_from_tmp_and_lbl(self):
        return self.get_captions("template") + self.get_captions("label")

    def get_labels(self):
        all_templates = sorted(set(self.annotations["template"]))
        print("Number of different classes: ", len(all_templates))
        class_dict = {k: idx for idx, k in enumerate(all_templates)}
        return [class_dict[p] for p in self.annotations["template"]]
    
    def get_samples_by_objects(self, objects):
        """
        :arg objects: list of objects of interest. e.g. ["bottle", "box"]
        :return: all samples in the dataset which contain at least one object
        from the specified list
        """

        inds_to_keep = ([i for (i, elem) in
                         enumerate(self.annotations["placeholders"])
                         if len(set(objects).intersection(elem)) > 0])
        return inds_to_keep
        
    def filter_annotations(self, objects):
        """
        :arg objects: list of objects that determines which samples to keep
        :return a subset of data where at least one of the objects are present.
        """

        inds_to_keep = self.get_samples_by_objects(objects)
        filtered_annotations = {}
        for field in self.annotations:
            filtered_annotations[field] = self.annotations[field][inds_to_keep]

        return filtered_annotations


class JsonParser(AnnotationParser):

    @classmethod
    def open_annotation(cls, path):
        if path.endswith("gz"):
            with gzip.open(path, "rb") as f:
                json = pd.read_json(f.read().decode("utf-8"))
        else:
            json = pd.read_json(path)
        return json

    def get_captions(self, caption_type=None):

        if caption_type is None:
            caption_type = self.caption_type

        if self.object_list:
            inds_to_keep_lbl = self.get_samples_by_objects(self.object_list)

            captions = []
            for i in range(len(self.annotations)):
                if i in inds_to_keep_lbl:
                    captions.append(self.annotations["label"][i])
                else:
                    captions.append(self.annotations["template"][i])

        else:
            print("Inside else: {}".format(caption_type))
            captions = [p for p in self.annotations[caption_type]]

        return captions

    def get_video_ids(self):

        ids = [id for id in self.annotations["id"]]
        return ids

    def get_video_paths(self):
        files = self.annotations[self.file_path]
        return [os.path.join(self.video_root, name.split("/")[0])
                for name in files]


class CSVParser(AnnotationParser):

    @classmethod
    def open_annotation(cls, path):
        if path.endswith("gz"):
            with gzip.open(path, "rb") as f:
                csv = pd.read_csv(f.read().decode("utf-8"))
        else:
            csv = pd.read_csv(path, delimiter=";",
                              names=["id", "template"])
        return csv

    def get_captions(self, caption_type=None):

        if caption_type is None:
            caption_type = self.caption_type

        captions = [i for i in self.annotations[caption_type]]

        return captions

    def get_video_ids(self):

        ids = [str(i)+".webm" for i in self.annotations["id"]]
        return ids


    def get_video_paths(self):
        return self.get_video_ids()


class V2Parser(JsonParser):

    def get_video_ids(self):

        ids = [str(i)+".webm" for i in self.annotations["id"]]
        return ids


    def get_video_paths(self):
        return [file for file in self.get_video_ids()]


class JsonV2Parser(JsonParser):

    def get_captions(self, caption_type=None):
        if caption_type is None:
            caption_type = self.caption_type
        return self.process_captions(caption_type)

    def process_captions(self, caption_type):
        if caption_type == "template":
            return self.annotations[caption_type].tolist()
        elif caption_type == "label":
            templates = self.annotations["template"]
            placeholders_series = self.annotations["placeholders"]
            labels = [self.sub(template, placeholders, self.single_object) for
                      template, placeholders in zip(templates,
                                                    placeholders_series)]
            return labels
        else:
            print("The input caption_type is not recognized")
            raise NotImplementedError

    @classmethod
    def sub(self, template, placeholders, single_object=False,
            pattern_tokens=("[", "]")):
        j = 0
        while any([True if sub in template else False for sub in pattern_tokens]):
            start = template.lower().find(pattern_tokens[0])
            end = template.lower().find(pattern_tokens[1])
            placeholder = (placeholders[j].split()[-1] if single_object
                           else placeholders[j])
            template = template[:start] + placeholder + template[end + 1:]
            j += 1
        return template
