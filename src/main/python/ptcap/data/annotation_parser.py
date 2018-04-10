import gzip
import os

import pandas as pd
from collections import OrderedDict

class_freqs = \
({'Approaching [something] with your camera': 1162,
         'Attaching [something] to [something]': 837,
         'Bending [something] so that it deforms': 497,
         'Bending [something] until it breaks': 487,
         'Burying [something] in [something]': 603,
         'Closing [something]': 1068,
         'Covering [something] with [something]': 2727,
         'Digging [something] out of [something]': 449,
         'Dropping [something] behind [something]': 783,
         'Dropping [something] in front of [something]': 890,
         'Dropping [something] into [something]': 903,
         'Dropping [something] next to [something]': 982,
         'Dropping [something] onto [something]': 1122,
         'Failing to put [something] into [something] because [something] does not fit': 273,
         'Folding [something]': 972,
         'Hitting [something] with [something]': 1738,
         'Holding [something]': 1459,
         'Holding [something] behind [something]': 991,
         'Holding [something] in front of [something]': 1781,
         'Holding [something] next to [something]': 1403,
         'Holding [something] over [something]': 1260,
         'Laying [something] on the table on its side, not upright': 689,
         'Letting [something] roll along a flat surface': 783,
         'Letting [something] roll down a slanted surface': 669,
         'Letting [something] roll up a slanted surface, so it rolls back down': 348,
         'Lifting [something] up completely without letting it drop down': 1673,
         'Lifting [something] up completely, then letting it drop down': 1593,
         'Lifting [something] with [something] on it': 1501,
         'Lifting a surface with [something] on it but not enough for it to slide down': 225,
         'Lifting a surface with [something] on it until it starts sliding down': 317,
         'Lifting up one end of [something] without letting it drop down': 1430,
         'Lifting up one end of [something], then letting it drop down': 1594,
         'Moving [part] of [something]': 639,
         'Moving [something] across a surface until it falls down': 687,
         'Moving [something] across a surface without it falling down': 680,
         'Moving [something] and [something] away from each other': 1520,
         'Moving [something] and [something] closer to each other': 1563,
         'Moving [something] and [something] so they collide with each other': 464,
         'Moving [something] and [something] so they pass each other': 501,
         'Moving [something] away from [something]': 910,
         'Moving [something] away from the camera': 881,
         'Moving [something] closer to [something]': 907,
         'Moving [something] down': 2741,
         'Moving [something] towards the camera': 853,
         'Moving [something] up': 3170,
         'Moving away from [something] with your camera': 1076,
         'Opening [something]': 1253,
         'Picking [something] up': 980,
         'Piling [something] up': 790,
         'Plugging [something] into [something]': 1195,
         'Plugging [something] into [something] but pulling it right out as you remove your hand': 671,
         'Poking [something] so it slightly moves': 1287,
         "Poking [something] so lightly that it doesn't or almost doesn't move": 2075,
         'Poking [something] so that it falls over': 679,
         'Poking [something] so that it spins around': 153,
         'Poking a hole into [some substance]': 91,
         'Poking a hole into [something soft]': 185,
         'Poking a stack of [something] so the stack collapses': 291,
         'Poking a stack of [something] without the stack collapsing': 217,
         'Pouring [something] into [something]': 873,
         'Pouring [something] into [something] until it overflows': 239,
         'Pouring [something] onto [something]': 274,
         'Pouring [something] out of [something]': 314,
         'Pretending or failing to wipe [something] off of [something]': 320,
         'Pretending or trying and failing to twist [something]': 311,
         'Pretending to be tearing [something that is not tearable]': 796,
         'Pretending to close [something] without actually closing it': 908,
         'Pretending to open [something] without actually opening it': 1473,
         'Pretending to pick [something] up': 1547,
         'Pretending to poke [something]': 523,
         'Pretending to pour [something] out of [something], but [something] is empty': 314,
         'Pretending to put [something] behind [something]': 697,
         'Pretending to put [something] into [something]': 1044,
         'Pretending to put [something] next to [something]': 1204,
         'Pretending to put [something] on a surface': 1391,
         'Pretending to put [something] onto [something]': 669,
         'Pretending to put [something] underneath [something]': 337,
         'Pretending to scoop [something] up with [something]': 322,
         'Pretending to spread air onto [something]': 169,
         'Pretending to sprinkle air onto [something]': 408,
         'Pretending to squeeze [something]': 717,
         'Pretending to take [something] from [somewhere]': 1204,
         'Pretending to take [something] out of [something]': 924,
         'Pretending to throw [something]': 915,
         'Pretending to turn [something] upside down': 725,
         'Pulling [something] from behind of [something]': 462,
         'Pulling [something] from left to right': 1555,
         'Pulling [something] from right to left': 1587,
         'Pulling [something] onto [something]': 290,
         'Pulling [something] out of [something]': 548,
         'Pulling two ends of [something] but nothing happens': 493,
         'Pulling two ends of [something] so that it gets stretched': 292,
         'Pulling two ends of [something] so that it separates into two pieces': 199,
         'Pushing [something] from left to right': 2949,
         'Pushing [something] from right to left': 2724,
         'Pushing [something] off of [something]': 526,
         'Pushing [something] onto [something]': 351,
         'Pushing [something] so it spins': 573,
         "Pushing [something] so that it almost falls off but doesn't": 962,
         'Pushing [something] so that it falls off the table': 1687,
         'Pushing [something] so that it slightly moves': 1874,
         'Pushing [something] with [something]': 1452,
         'Putting [number of] [something] onto [something]': 955,
         'Putting [something similar to other things that are already on the table]': 1766,
         'Putting [something that cannot actually stand upright] upright on the table, so it falls on its side': 591,
         'Putting [something] and [something] on the table': 1103,
         'Putting [something] behind [something]': 1204,
         'Putting [something] in front of [something]': 837,
         'Putting [something] into [something]': 2188,
         'Putting [something] next to [something]': 2031,
         'Putting [something] on a flat surface without letting it roll': 396,
         'Putting [something] on a surface': 3284,
         'Putting [something] on the edge of [something] so it is not supported and falls down': 507,
         'Putting [something] onto [something else that cannot support it] so it falls down': 364,
         'Putting [something] onto [something]': 1608,
         "Putting [something] onto a slanted surface but it doesn't glide down": 156,
         "Putting [something] that can't roll onto a slanted surface, so it slides down": 368,
         "Putting [something] that can't roll onto a slanted surface, so it stays where it is": 361,
         'Putting [something] underneath [something]': 609,
         'Putting [something] upright on the table': 733,
         'Putting [something], [something] and [something] on the table': 982,
         'Removing [something], revealing [something] behind': 903,
         'Rolling [something] on a flat surface': 1255,
         'Scooping [something] up with [something]': 936,
         'Showing [something] behind [something]': 1678,
         'Showing [something] next to [something]': 832,
         'Showing [something] on top of [something]': 922,
         'Showing [something] to the camera': 709,
         'Showing a photo of [something] to the camera': 837,
         'Showing that [something] is empty': 1638,
         'Showing that [something] is inside [something]': 1213,
         'Spilling [something] behind [something]': 111,
         'Spilling [something] next to [something]': 162,
         'Spilling [something] onto [something]': 266,
         'Spinning [something] so it continues spinning': 747,
         'Spinning [something] that quickly stops spinning': 1071,
         'Spreading [something] onto [something]': 415,
         'Sprinkling [something] onto [something]': 400,
         'Squeezing [something]': 1936,
         'Stacking [number of] [something]': 1032,
         'Stuffing [something] into [something]': 1379,
         'Taking [one of many similar things on the table]': 2275,
         'Taking [something] from [somewhere]': 1032,
         'Taking [something] out of [something]': 1699,
         'Tearing [something] into two pieces': 1736,
         'Tearing [something] just a little bit': 1285,
         'Throwing [something]': 2254,
         'Throwing [something] against [something]': 1254,
         'Throwing [something] in the air and catching it': 943,
         'Throwing [something] in the air and letting it fall': 774,
         'Throwing [something] onto a surface': 875,
         "Tilting [something] with [something] on it slightly so it doesn't fall down": 604,
         'Tilting [something] with [something] on it until it falls off': 928,
         'Tipping [something] over': 441,
         'Tipping [something] with [something in it] over, so [something in it] falls out': 326,
         'Touching (without moving) [part] of [something]': 1471,
         "Trying but failing to attach [something] to [something] because it doesn't stick": 471,
         'Trying to bend [something unbendable] so nothing happens': 748,
         'Trying to pour [something] into [something], but missing so it spills next to it': 183,
         'Turning [something] upside down': 2058,
         'Turning the camera downwards while filming [something]': 867,
         'Turning the camera left while filming [something]': 1029,
         'Turning the camera right while filming [something]': 1018,
         'Turning the camera upwards while filming [something]': 851,
         'Twisting (wringing) [something] wet until water comes out': 240,
         'Twisting [something]': 724,
         'Uncovering [something]': 2426,
         'Unfolding [something]': 840,
         'Wiping [something] off of [something]': 539,
         '[Something] being deflected from [something]': 327,
         '[Something] colliding with [something] and both are being deflected': 462,
         '[Something] colliding with [something] and both come to a halt': 448,
         '[Something] falling like a feather or paper': 1100,
         '[Something] falling like a rock': 1390})

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

        class_dict = {k: idx for idx, k in enumerate(all_templates)}
        return [class_dict[p] for p in self.annotations["template"]]

        return [class_dict[action_groups[p.replace("[","").replace("]","")]]
                for p in self.annotations["template"]]


    def get_class_pop(self):
        templates = self.annotations["template"]
        from collections import Counter
        class_pop = Counter(templates)
        return class_pop
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



