import json
import gzip

def json_process(path):

    new_json = []
    all_samples = open_json(path)
    for sample in all_samples:
        template = sample["template"]
        placeholders = sample["placeholders"]

        label = template
        for p in placeholders:
            ind_begin = label.find("[")
            ind_end = label.find("]")

            label = label[:ind_begin]+p+label[ind_end+1:]
        print(label)
        del sample["label"]
        sample["label"] = label
        new_json.append(sample)

    return new_json

def open_json(path):
    if path.endswith("gz"):
        with gzip.open(path, "rb") as f:
            loaded_json = json.loads(f.read().decode("utf-8"))
    else:
        with open(path) as f:
            loaded_json = json.load(f)
    return loaded_json


if __name__ == "__main__":
    input_jsons = ["/home/farzaneh/PycharmProjects/pytorch-captioning/src/main/python/scripts/something-something-v2-train.json",
                   "/home/farzaneh/PycharmProjects/pytorch-captioning/src/main/python/scripts/something-something-v2-validation.json"]
    output_jsons = [
        "/home/farzaneh/PycharmProjects/pytorch-captioning/src/main/python/scripts/v2-train.json",
        "/home/farzaneh/PycharmProjects/pytorch-captioning/src/main/python/scripts/v2-validation.json"]


    for in_json, out_json in zip(input_jsons, output_jsons):
        print("Creating subset from {} to {}".format(in_json, out_json))
        new_json = json_process(in_json)

        with open(out_json, "w") as f:
            json.dump(new_json, f)
