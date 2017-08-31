import os
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skvideo.io import vwrite, ffprobe

TMP_DIR = "fake_data"
VIDEOS_DIR = os.path.join(TMP_DIR, "videos")
FRAMES_DIR = os.path.join(TMP_DIR, "frames")
JSON_DIR = os.path.join(TMP_DIR, "json")
JSON_FILE = os.path.join(JSON_DIR, "fake.json")
CSV_FILE = os.path.join(JSON_DIR, "fake.csv")
MP4_FILENAME = 'ANY_FILE.mp4'
VIDEO_SIZE = [20, 40, 40]
CROP_SIZE = [8, 24, 24]
NUM_CLASSES = 5
NUM_SAMPLES = 10
VIDEO_IDS = [i % NUM_CLASSES for i in range(NUM_SAMPLES)]


def create_fake_json(durations):
    '''Inspired from https://github.com/TwentyBN/20bn-realtimenet/blob/master/smoke_test.py
    Credits to Ingo F.'''
    json_content = [{
        "id": i,
        "file": "{}/{}".format(i, MP4_FILENAME),
        "duration": "%.6f" % durations[i],
        "width": VIDEO_SIZE[1],
        "height": VIDEO_SIZE[2],
        "label": "ANY LABEL {}".format(int(i) % NUM_CLASSES),
        "template": "ANY LABEL {}".format(int(i) % NUM_CLASSES),
        "worker_id": "ANY_WORKER",
    } for i in VIDEO_IDS]

    with open(JSON_FILE, 'w') as f:
        json.dump(json_content, f)


def create_fake_csv():
    csv_content = ['%d/ANY_FILE.mp4\n' % i for i in range(5)]
    with open(CSV_FILE, 'w') as f:
        f.writelines(csv_content)


def create_fake_video_data():
    # Create subfolders
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    # Create each subdir
    videos_subdirs = [os.path.join(VIDEOS_DIR, str(id)) for id in VIDEO_IDS]
    frames_subdirs = [os.path.join(FRAMES_DIR, str(id)) for id in VIDEO_IDS]

    # Empty duration list
    durations = []

    # Create video files
    for video_subdir, frame_subdir in zip(videos_subdirs, frames_subdirs):
        # Create subdir
        os.makedirs(video_subdir, exist_ok=True)
        os.makedirs(frame_subdir, exist_ok=True)
        # Create mpeg file
        video_path = os.path.join(video_subdir, MP4_FILENAME)
        video = create_one_fake_video(VIDEO_SIZE)
        vwrite(video_path, video)
        # Get duration
        duration = float(ffprobe(video_path)['video']['@duration'])
        durations.append(duration)
        # Create jpeg frames
        for i, frame in enumerate(video):
            plt.imsave(os.path.join(frame_subdir, str(i) + ".jpg"), frame)

    # Create annotation files
    create_fake_json(durations)
    create_fake_csv()


def create_one_fake_video(size):
    return np.random.randint(0, 255, size[0] * size[1] * size[2] * 3).reshape((size[0], size[1], size[1], 3))


def remove_fake_data():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)