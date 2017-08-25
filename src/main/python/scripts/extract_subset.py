import json
import gzip

NUM_SAMPLES = 1000

TARGET_LABELS = [
    "Pretending to open [something] without actually opening it",
    "Opening [something]",
]



def create_subset_json(path, target_classes, num_samples=None):
    """
    This method extracts 'num_samples' samples using a json annotation file
    where each sample belong to one of the classes in target_classes.
    If num_samples is None, it extracts all the samples in target_classes.
    """

    if num_samples:
        assert (num_samples > 0)
    assert (len(target_classes) > 1)

    counter = 0
    new_json = []
    all_samples = open_json(path)
    for sample in all_samples:
        if sample['template'] in target_classes:
            if not num_samples or counter < num_samples:
                new_json.append(sample)
                counter += 1

    return new_json


def open_json(path):
    if path.endswith("gz"):
        f = gzip.open(path, "rb")
        loaded_json = json.loads(f.read().decode("utf-8"))
    else:
        f = open(path)
        loaded_json = json.load(f)
    return loaded_json


if __name__ == "__main__":
    input_jsons = ["/data/20bn-objects/json/train_20170429.json.gz",
                   "/data/20bn-objects/json/validation_20170429.json.gz",
                   "/data/20bn-objects/json/test_20170429.json.gz"]
    output_jsons = ["subset_train_20170429.json.gz",
                    "subset_validation_20170429.json.gz",
                    "subset_test_20170429.json.gz"]

    print('Number of labels in subset dataset : {}'.format(len(TARGET_LABELS)))

    for in_json, out_json in zip(input_jsons, output_jsons):
        print('Creating subset from {} to {}'.format(in_json, out_json))
        new_json = create_subset_json(in_json, TARGET_LABELS, num_samples=10000)

        with gzip.open(out_json, 'wt') as f:
            json.dump(new_json, f)
        with gzip.open("subset_classes.json.gz", 'wt') as f:
            json.dump(TARGET_LABELS, f)
