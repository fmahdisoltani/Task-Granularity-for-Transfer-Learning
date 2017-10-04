# Code copied from 20bn-rtorchn repo

import json
import os
import shutil


import numpy as np

TMP_DIR = "fake_data"
VIDEOS_DIR = os.path.join(TMP_DIR, "videos")
JSON_DIR = os.path.join(TMP_DIR, "json")
CSV_FILE = os.path.join(JSON_DIR, "fake.csv")
JSON_FILE = os.path.join(JSON_DIR, "fake.json")

CROP_SIZE = [8, 24, 24]
MP4_FILENAME = 'ANY_FILE.mp4'
NUM_CLASSES = 2
NUM_SAMPLES = 3
VIDEO_LABELS = [i % NUM_CLASSES for i in range(NUM_SAMPLES)]
VIDEO_SIZE = [20, 40, 40]



def create_fake_json(path=None):
    """
        Inspired from https://github.com/TwentyBN/20bn-realtimenet/blob/master/smoke_test.py
        Credits to Ingo F.
    """

    if path is None:
        path = JSON_FILE

    json_content = [{
        "id": i,
        "file": "{}/{}".format(i, MP4_FILENAME),
        "width": VIDEO_SIZE[1],
        "height": VIDEO_SIZE[2],
        "label": "ANY LABEL {}".format(VIDEO_LABELS[i]),
        "template": "ANY LABEL {}".format(VIDEO_LABELS[i]),
        "worker_id": "ANY_WORKER",
    } for i in range(NUM_SAMPLES)]

    with open(path, 'w') as f:
        json.dump(json_content, f)


def create_fake_video_data():
    # Create subfolders
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    # Create video subdir
    videos_subdirs = [os.path.join(VIDEOS_DIR, str(id_)) for id_ in
                      range(NUM_SAMPLES)]

    # Create video files
    for video_subdir in videos_subdirs:
        # Create subdir
        os.makedirs(video_subdir, exist_ok=True)
        # Create mpeg file
        video_path = os.path.join(video_subdir, MP4_FILENAME)
        video = create_one_fake_video(VIDEO_SIZE)
        np.savez_compressed(video_path, video)

    # Create annotation files
    create_fake_json()


def create_one_fake_video(size):
    return np.random.randint(0, 255, size[0] * size[1] * size[2] * 3
                             ).reshape((size[0], size[1], size[1], 3))


def remove_dir(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
