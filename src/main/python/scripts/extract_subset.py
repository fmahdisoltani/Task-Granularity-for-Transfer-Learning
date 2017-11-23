import json
import gzip

NUM_SAMPLES = 6000

TARGET_LABELS = [
    "Opening [something]",
    "Tearing [something] into two pieces",
]


def create_subset_json(path, target_classes, num_samples=None):
    """
    This method extracts a total of "num_samples" samples using a json
    annotation file where each sample belong to one of the classes in
    target_classes. If num_samples is None, it extracts all the samples in
    target_classes.
    """

    if num_samples:
        assert (num_samples > 0)
    assert (len(target_classes) > 1)

    counter = 0
    new_json = []
    all_samples = open_json(path)
    for sample in all_samples:
        if sample["template"] in target_classes:
            if num_samples is None or counter < num_samples:
                new_json.append(sample)
                counter += 1

    return new_json


def create_subset_json_balanced(path, target_classes, num_samples=None):
    """
    This method extracts at most "num_samples" samples using a json annotation
    file where each sample belong to one of the classes in target_classes.
    If num_samples is None, it extracts all the samples in target_classes.
    """

    num_classes = len(target_classes)
    if num_samples:
        assert (num_samples > 0)
    assert (num_classes > 1)

    population_dict = {k: 0 for k in target_classes}
    finished_classes = 0
    new_json = []
    all_samples = open_json(path)

    for sample in all_samples:
        if finished_classes == num_classes: break
        if sample["template"] in target_classes:
            if population_dict[sample["template"]] < num_samples:
                new_json.append(sample)
                population_dict[sample["template"]] += 1
                if population_dict[sample["template"]] == num_samples:
                    finished_classes += 1

                    print(sample["template"])
                    print(population_dict[sample["template"]])

    print(len(new_json))

    return new_json


def create_subset_json_balanced2(path, target_classes, num_samples=None):
    """
    This method is the result of a "Challenge Accepted" to implement the above
    function in only line
    """

    return [item for sublist in [[x for x in open_json(path) if x == t][:num_samples]
                                 for t in [sample for sample in json if
                      sample['template'] in target_classes]] for item in sublist]

def open_json(path):
    if path.endswith("gz"):
        with gzip.open(path, "rb") as f:
            loaded_json = json.loads(f.read().decode("utf-8"))
    else:
        with open(path) as f:
            loaded_json = json.load(f)
    return loaded_json


if __name__ == "__main__":


    input_jsons = ["Users/farzaneh/PycharmProjects/TwentyBN/pytorch-captioning/sample_data/json/train_20170320.json"]

    output_jsons = ["Users/farzaneh/PycharmProjects/TwentyBN/pytorch-captioning/sample_data/jsom/subset_train_20170320.json.gz"]

    print("Number of labels in subset dataset : {}".format(len(TARGET_LABELS)))

    for in_json, out_json in zip(input_jsons, output_jsons):
        print("Creating subset from {} to {}".format(in_json, out_json))
        new_json = create_subset_json_balanced(in_json, TARGET_LABELS,
                                               num_samples=10)

        with gzip.open(out_json, "wt") as f:
            json.dump(new_json, f)
        with gzip.open("subset_classes.json.gz", "wt") as f:
            json.dump(TARGET_LABELS, f)
