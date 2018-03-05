from ptcap.main import train_model
from ptcap.data.annotation_parser import JsonV2Parser
from ptcap.data.config_parser import YamlConfig

import os
import pickle


def decode(valid_captions, tokenizer, file_name):

    decoded_captions = []
    for caption_batch in valid_captions:
        np_caption_batch = caption_batch.cpu().data.numpy()
        for caption in np_caption_batch:
            decoded_captions.append(tokenizer.get_string(caption))
    with open(file_name, 'wb') as f:
        pickle.dump(decoded_captions, f)


if __name__ == '__main__':

    json_train = "/home/waseem/data/json/something-something-v2-train.json"
    json_valid = "/home/waseem/data/json/something-something-v2-validation.json"
    json_test = "/home/waseem/data/json/something-something-v2-test.json"

    json_paths = [json_train, json_valid, json_test]

    v2parser_train = JsonV2Parser(json_train, None, single_object=True)
    v2parser_valid = JsonV2Parser(json_valid, None, single_object=True)
    v2parser_test = JsonV2Parser(json_test, None, single_object=True)

    training_templates = v2parser_train.get_captions("template")
    training_labels = v2parser_train.get_captions("label")

    templatexlabel_count_dict = {}
    templatexlabel_max = {}
    for template, label in zip(training_templates, training_labels):
        if template not in templatexlabel_count_dict:
            templatexlabel_count_dict[template] = {}
            templatexlabel_max[template] = [None, 0]
        label_count_dict = templatexlabel_count_dict[template]
        if label not in label_count_dict:
            label_count_dict[label] = 1
        else:
            label_count_dict[label] += 1
        if label_count_dict[label] > templatexlabel_max[template][1]:
            templatexlabel_max[template] = [label, label_count_dict[label]]

    parent_folder = "/home/waseem/Models/v2_gulp160_two_stream_c2_32_c3_32_labels_cutoff5_cassif0.1_cap0.9_simple/"

    config_path = os.path.join(parent_folder, "config1.yaml")
    valid_captions, valid_preds, tokenizer = train_model(YamlConfig(config_path))

    decode(valid_captions, tokenizer, os.path.join(parent_folder, "decoded_captions_no_tf"))
    decode(valid_preds, tokenizer, os.path.join(parent_folder, "decoded_preds_no_tf"))

    print("Script complete!")
