import json
import gzip
import pandas as pd

from ptcap.data.annotation_parser import JsonParser

NUM_SAMPLES = 1000

TARGET_LABELS = [

    "Pretending to open [something] without actually opening it",
    "Opening [something]",

]

TEMP=["Tearing [something] just a little bit",
"Turning the camera left while filming [something]",
    "Pulling [something] from right to left",
    "Burying [something] in [something]",
    "Pretending to pick [something] up",
    "Scooping [something] up with [something]",
    "Moving [something] up",
    "Pushing [something] with [something]",
    "Spinning [something] that quickly stops spinning",
    "Uncovering [something]",
    "Showing that [something] is inside [something]",
    "Moving [something] away from [something]",
    "Bending [something] so that it deforms",
    "Closing [something]",
    "Showing that [something] is inside [something]",
    "Dropping [something] onto [something]",
    "Picking [something] up",
    "[Something] falling like a feather or paper",
    "Taking [something] out of [something]",
    "Dropping [something] into [something]",
    "Turning the camera left while filming [something]",
    "Pouring [something] into [something]",
    "Pretending to turn [something] upside down",
    "Pulling [something] out of [something]",
    "Spilling [something] onto [something]",
    "[Something] colliding with [something] and both are being deflected",
    "[Something] being deflected from [something]",
    "Dropping [something] in front of [something]",
    "Pretending to close [something] without actually closing it",
    "Opening [something]",
    "Covering [something] with [something]",
    "Attaching [something] to [something]",
    "Taking [something] out of [something]",
    "[Something] falling like a rock",
    "Closing [something]"

]


def create_subset_json(path, num_samples, target_labels):
    """
    This method extracts samples from a json annotation file where the sample
    belongs to a class in target_labels
    """

    counter = 0
    new_json = []
    with gzip.open(path, 'rt') as fp:
        loaded_json = json.load(fp)
        for i in loaded_json:
            # print("\"" + i['template'] + "\",")
            if i['template'] in target_labels:
                #if counter < num_samples and
                    new_json.append(i)
                    counter += 1
    return new_json


if __name__ == "__main__":
    input_jsons = ["/data/20bn-objects/json/train_20170429.json.gz",
                   "/data/20bn-objects/json/validation_20170429.json.gz",
                   "/data/20bn-objects/json/test_20170429.json.gz"]
    output_jsons = ["subset_train_20170429.json.gz",
                    "subset_validation_20170429.json.gz",
                    "subset_test_20170429.json.gz"]

    print('numer of labels in subset dataset : {}'.format(len(TARGET_LABELS)))

    for i, annot in enumerate(input_jsons):
        print('converting {} to {}'.format(input_jsons[i], output_jsons[i]))
        print("^" * 100)
        print(annot)
        new_json = create_subset_json(input_jsons[i], 10000, TARGET_LABELS)
        with gzip.open(output_jsons[i], 'wt') as f:
            json.dump(new_json, f)
        with gzip.open("subset_classes.json.gz", 'wt') as f:
            json.dump(TARGET_LABELS, f)

