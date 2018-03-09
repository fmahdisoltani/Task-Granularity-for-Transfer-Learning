import gzip
import os

import pandas as pd
from collections import OrderedDict
action_groups=\
    OrderedDict([('Attaching something to something', 'Attaching/Trying to attach'),
("Trying but failing to attach something to something because it doesn't stick",
 'Attaching/Trying to attach'),
('Bending something so that it deforms', 'Bending something'),
('Bending something until it breaks', 'Bending something'),
('Trying to bend something unbendable so nothing happens',
 'Bending something'),
('Digging something out of something', 'Burying or digging something'),
('Burying something in something', 'Burying or digging something'),
('Turning the camera downwards while filming something', 'Camera motions'),
('Moving something towards the camera', 'Camera motions'),
('Turning the camera left while filming something', 'Camera motions'),
('Moving away from something with your camera', 'Camera motions'),
('Moving something away from the camera', 'Camera motions'),
('Turning the camera upwards while filming something', 'Camera motions'),
('Approaching something with your camera', 'Camera motions'),
('Turning the camera right while filming something', 'Camera motions'),
('Something being deflected from something', 'Collisions of objects'),
('Something colliding with something and both come to a halt',
 'Collisions of objects'),
('Something colliding with something and both are being deflected',
 'Collisions of objects'),
('Covering something with something', 'Covering'),
('Uncovering something', 'Covering'),
('Putting something similar to other things that are already on the table',
 'Crowd of things'),
('Taking one of many similar things on the table', 'Crowd of things'),
('Dropping something next to something', 'Dropping something'),
('Dropping something onto something', 'Dropping something'),
('Dropping something behind something', 'Dropping something'),
('Dropping something into something', 'Dropping something'),
('Dropping something in front of something', 'Dropping something'),
('Showing something on top of something',
 'Filming objects, without any actions'),
('Showing something behind something',
 'Filming objects, without any actions'),
('Showing something next to something',
 'Filming objects, without any actions'),
('Unfolding something', 'Folding something'),
('Folding something', 'Folding something'),
('Hitting something with something', 'Hitting something with something'),
('Holding something in front of something', 'Holding something'),
('Holding something over something', 'Holding something'),
('Holding something behind something', 'Holding something'),
('Holding something next to something', 'Holding something'),
('Holding something', 'Holding something'),
('Lifting up one end of something, then letting it drop down',
 'Lifting and (not) dropping something'),
('Lifting something up completely, then letting it drop down',
 'Lifting and (not) dropping something'),
('Lifting up one end of something without letting it drop down',
 'Lifting and (not) dropping something'),
('Lifting something up completely without letting it drop down',
 'Lifting and (not) dropping something'),
('Tilting something with something on it until it falls off',
 'Lifting/Tilting objects with other objects on them'),
('Lifting something with something on it',
 'Lifting/Tilting objects with other objects on them'),
("Tilting something with something on it slightly so it doesn't fall down",
 'Lifting/Tilting objects with other objects on them'),
('Moving something down', 'Moving something'),
('Moving something up', 'Moving something'),
('Moving something and something away from each other',
 'Moving two objects relative to each other'),
('Moving something closer to something',
 'Moving two objects relative to each other'),
('Moving something away from something',
 'Moving two objects relative to each other'),
('Moving something and something closer to each other',
 'Moving two objects relative to each other'),
('Moving part of something', 'Moving/Touching a part of something'),
('Touching (without moving) part of something',
 'Moving/Touching a part of something'),
('Pretending to close something without actually closing it',
 'Opening or closing something'),
('Closing something', 'Opening or closing something'),
('Opening something', 'Opening or closing something'),
('Pretending to open something without actually opening it',
 'Opening or closing something'),
('Picking something up', 'Picking something up'),
('Pretending to pick something up', 'Picking something up'),
('Piling something up', 'Piles of stuff'),
('Plugging something into something', 'Plugging something into something'),
('Plugging something into something but pulling it right out as you remove your hand',
 'Plugging something into something'),
("Poking something so lightly that it doesn't or almost doesn't move",
 'Poking something'),
('Poking a stack of something so the stack collapses', 'Poking something'),
('Poking something so that it falls over', 'Poking something'),
('Poking a hole into some substance', 'Poking something'),
('Poking a hole into something soft', 'Poking something'),
('Poking something so that it spins around', 'Poking something'),
('Poking a stack of something without the stack collapsing',
 'Poking something'),
('Pretending to poke something', 'Poking something'),
('Poking something so it slightly moves', 'Poking something'),
('Pouring something into something', 'Pouring something'),
('Pouring something into something until it overflows', 'Pouring something'),
('Pretending to pour something out of something, but something is empty',
 'Pouring something'),
('Pouring something out of something', 'Pouring something'),
('Trying to pour something into something, but missing so it spills next to it',
 'Pouring something'),
('Pouring something onto something', 'Pouring something'),
('Pulling something from right to left', 'Pulling something'),
('Pulling something from left to right', 'Pulling something'),
('Pulling something onto something', 'Pulling something'),
('Pulling something out of something', 'Pulling something'),
('Pulling something from behind of something', 'Pulling something'),
('Pulling two ends of something but nothing happens',
 'Pulling two ends of something'),
('Pulling two ends of something so that it separates into two pieces',
 'Pulling two ends of something'),
('Pulling two ends of something so that it gets stretched',
 'Pulling two ends of something'),
('Pushing something from left to right', 'Pushing something'),
('Pushing something so that it slightly moves', 'Pushing something'),
('Pushing something so that it falls off the table', 'Pushing something'),
("Pushing something so that it almost falls off but doesn't",
 'Pushing something'),
('Pushing something with something', 'Pushing something'),
('Pushing something off of something', 'Pushing something'),
('Pushing something onto something', 'Pushing something'),
('Pushing something from right to left', 'Pushing something'),
('Pretending to put something on a surface', 'Putting something somewhere'),
('Putting something on a surface', 'Putting something somewhere'),
('Laying something on the table on its side, not upright',
 'Putting something upright/on its side'),
('Putting something that cannot actually stand upright upright on the table, so it falls on its side',
 'Putting something upright/on its side'),
('Putting something upright on the table',
 'Putting something upright/on its side'),
('Pretending to put something behind something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Failing to put something into something because something does not fit',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Pretending to put something next to something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Pretending to put something onto something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Pretending to put something into something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something behind something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something, something and something on the table',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something onto something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Taking something out of something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something and something on the table',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something next to something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something in front of something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something into something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something underneath something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something onto something else that cannot support it so it falls down',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Pretending to take something out of something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Pretending to put something underneath something',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Putting something on the edge of something so it is not supported and falls down',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Removing something, revealing something behind',
 'Putting/Taking objects into/out of/next to/… other objects'),
('Rolling something on a flat surface', 'Rolling and sliding something'),
('Letting something roll up a slanted surface, so it rolls back down',
 'Rolling and sliding something'),
('Letting something roll down a slanted surface',
 'Rolling and sliding something'),
('Lifting a surface with something on it until it starts sliding down',
 'Rolling and sliding something'),
('Putting something on a flat surface without letting it roll',
 'Rolling and sliding something'),
('Lifting a surface with something on it but not enough for it to slide down',
 'Rolling and sliding something'),
("Putting something onto a slanted surface but it doesn't glide down",
 'Rolling and sliding something'),
("Putting something that can't roll onto a slanted surface, so it stays where it is",
 'Rolling and sliding something'),
("Putting something that can't roll onto a slanted surface, so it slides down",
 'Rolling and sliding something'),
('Letting something roll along a flat surface',
 'Rolling and sliding something'),
('Scooping something up with something', 'Scooping something up'),
('Pretending to scoop something up with something', 'Scooping something up'),
('Showing a photo of something to the camera',
 'Showing objects and photos of objects'),
('Showing something to the camera', 'Showing objects and photos of objects'),
('Showing that something is inside something',
 'Showing that something is full/empty'),
('Showing that something is empty', 'Showing that something is full/empty'),
('Moving something across a surface without it falling down',
 'Something (not) falling over an edge'),
('Moving something across a surface until it falls down',
 'Something (not) falling over an edge'),
('Something falling like a rock', 'Something falling'),
('Something falling like a feather or paper', 'Something falling'),
('Moving something and something so they collide with each other',
 'Something passing/hitting another thing'),
('Moving something and something so they pass each other',
 'Something passing/hitting another thing'),
('Spilling something onto something', 'Spilling something'),
('Spilling something next to something', 'Spilling something'),
('Spilling something behind something', 'Spilling something'),
('Pushing something so it spins', 'Spinning something'),
('Spinning something so it continues spinning', 'Spinning something'),
('Spinning something that quickly stops spinning', 'Spinning something'),
('Spreading something onto something', 'Spreading something onto something'),
('Pretending to spread air onto something',
 'Spreading something onto something'),
('Pretending to sprinkle air onto something',
 'Sprinkling something onto something'),
('Sprinkling something onto something',
 'Sprinkling something onto something'),
('Pretending to squeeze something', 'Squeezing something'),
('Squeezing something', 'Squeezing something'),
('Putting number of something onto something',
 'Stacking or placing N things'),
('Stacking number of something', 'Stacking or placing N things'),
('Stuffing something into something', 'Stuffing/Taking out'),
('Taking something from somewhere', 'Taking something'),
('Pretending to take something from somewhere', 'Taking something'),
('Pretending to be tearing something that is not tearable',
 'Tearing something'),
('Tearing something just a little bit', 'Tearing something'),
('Tearing something into two pieces', 'Tearing something'),
('Throwing something against something', 'Throwing something'),
('Throwing something', 'Throwing something'),
('Pretending to throw something', 'Throwing something'),
('Throwing something onto a surface', 'Throwing something'),
('Throwing something in the air and catching it', 'Throwing something'),
('Throwing something in the air and letting it fall', 'Throwing something'),
('Tipping something over', 'Tipping something over'),
('Tipping something with something in it over, so something in it falls out',
 'Tipping something over'),
('Pretending to turn something upside down', 'Turning something upside down'),
('Turning something upside down', 'Turning something upside down'),
('Twisting (wringing) something wet until water comes out',
 'Twisting something'),
('Twisting something', 'Twisting something'),
('Pretending or trying and failing to twist something', 'Twisting something'),
('Wiping something off of something', 'Wiping something off of something'),
('Pretending or failing to wipe something off of something',
 'Wiping something off of something')])

class AnnotationParser(object):

    def __init__(self, annot_path, video_root,
                 file_path="file", caption_type="template", object_list=None):
        self.video_root = video_root
        self.file_path = file_path
        self.caption_type = caption_type
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
        all_actions = sorted(set(action_groups.values()))
        print("Number of different classes: ", len(all_templates))
        print("Number of different action groups: ", len(all_actions))

        class_dict = {k: idx for idx, k in enumerate(all_actions)}
        return [class_dict[action_groups[p.replace("[","").replace("]","")]]
                for p in self.annotations["template"]]

    
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



