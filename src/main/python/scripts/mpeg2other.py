"""Input pipeline benchmark script.

Usage:
  videos2numpy.py <path_in>  <path_out>  --f=<framerate>  --s=<size>  --n=<nb_workers>  --fmt=<format> [--overwrite]
  videos2numpy.py (-h | --help)

Options:
  <path_in>          Path to a video dataset (e.g. /data/20bn-gestures/videos)
  <path_out>         Where to save videos (e.g. /data/20bn-gestures/videos_128x128)
  --f=<framerate>    Framerate.
  --s=<size>         Frame size.
  --n=<nb_workers>   Nb of workers.
  --fmt=<format>     Format file of output files.
  -h --help          Show this screen.
"""

import os
import gzip
import time

import pandas as pd
import bloscpack as bp
import numpy as np
from docopt import docopt
from skvideo.io import ffprobe
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skvideo.io import FFmpegReader

#from rtorchn.data import MpegVideoDataset

def string_to_one_hot(label, dictionary, dtype="float32"):
    out = np.zeros(len(dictionary), dtype=dtype)
    out[dictionary[label]] = 1
    return out

class VideoDataset(Dataset):
    def __init__(self, root, json_file, framerate=25, size=(224, 224), preprocessing=None,
                 label_shape=None):
        self.root = root
        # Read json file
        self.open_json(json_file)
        # Get file paths
        files = list(self.json.file)
        self.files = [os.path.join(root, f) for f in files]
        # Get durations
        self.durations = list(self.json.duration)
        # Get labels
        self.labels = list(self.json.label)
        # Get label2int dictionnary
        unique_classes = sorted(set(self.labels))
        self.labels2int = dict(zip(unique_classes, np.arange(len(unique_classes))))
        # Preprocessing
        self.N = len(self.files)
        self.framerate = int(framerate)
        self.size = [size[0], size[1]]
        self.preprocessing = preprocessing
        self.label_shape = label_shape

    def open_json(self, path):
        if path.endswith("gz"):
            with gzip.open(path, "rb") as f:
                self.json = pd.read_json(f.read().decode("ascii"))
        else:
            self.json = pd.read_json(path)

    def __getitem__(self, index):
        # Get data
        try:
            video = self.open_video(index)
        except:
            print("\nOpening", self.files[index], "failed.")
            video = np.zeros((100, self.size[0], self.size[1], 3))
        label = self.labels[index]
        # Preprocessing
        if self.preprocessing is not None:
            video = self.preprocessing.apply(video)
        label = string_to_one_hot(label, self.labels2int)
        if self.label_shape is not None:
            label = label.reshape(self.label_shape)
        # Return the sample | target pair
        return video, label

    def __len__(self):
        return self.N

    def open_video(self, index):
        pass


class MpegVideoDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super(MpegVideoDataset, self).__init__(*args, **kwargs)

    def open_video(self, index):
        # Get path and duration
        duration = self.durations[index]
        path = self.files[index]
        # Compute corresponding nb of frames
        nframes = int(duration * self.framerate)
        oargs = {"-r": "%d" % self.framerate,
                 "-vframes": "%d" % nframes}
        if self.size[0] and self.size[1]:
            oargs["-s"] = "%dx%d" % (self.size[0], self.size[1])
        # Open file
        reader = FFmpegReader(path, inputdict={}, outputdict=oargs)
        video = []
        # Get frames until there is no more
        for frame in reader.nextFrame():
            video.append(frame)
        # Return as a numpy array
        return np.array(video)


class VideoDatasetWriter(MpegVideoDataset):
    def __init__(self, path, path_out, framerate, size, format="blp", overwrite=False):
        # Path to video files
        self.path = path
        self.path_out = path_out
        self.get_path_to_videos()
        # Parameters for FFMpeg reading
        self.N = len(self.files)
        self.framerate = int(framerate)
        self.size = [size[0], size[1]]
        # Overwrite existing output file
        self.overwrite = overwrite
        self.format = format

    def get_path_to_videos(self):
        if not os.path.exists(self.path_out):
            os.mkdir(self.path_out)
        # Get path to subdirs
        dirnames = os.listdir(self.path)
        self.files = [os.path.join(self.path, f) for f in dirnames]
        self.files_out = [os.path.join(self.path_out, f) for f in dirnames]
        # Make output subdirs
        for k, subdir in enumerate(self.files_out):
            if not os.path.exists(subdir):
                os.mkdir(subdir)
                print("New folder created:", subdir)
        # Add video basenames
        self.durations = []
        for k, f, f_out in zip(range(len(self.files)), self.files, self.files_out):
            name = os.listdir(f)[0]
            self.files_out[k] = os.path.join(f_out, name)
            self.files[k] = os.path.join(f, name)
            self.durations.append(self.get_duration(self.files[k]))

    def __getitem__(self, index):
        if not os.path.exists(self.files_out[index]) or self.overwrite:
            try:
                video = self.open_video(index)
            except:
                print("\nOpening", self.files[index], "failed.")
                video = np.zeros((int(self.durations[index] * self.framerate), self.size[0], self.size[1], 3))
            # save video
            self.save_video(self.files_out[index], video)
        return np.array([1])

    def get_duration(self, file):
        metadata = ffprobe(file)
        return float(metadata['video']['@duration'])

    def save_video(self, path, video):
        if self.format == "npy":
            np.savez_compressed(path, video)
        else:
            bp.pack_ndarray_file(video, path + ".blp")


if __name__ == '__main__':
    # Get argument
    args = docopt(__doc__)

    # Parse args
    path_in = args['<path_in>']
    path_out = args['<path_out>']
    framerate = int(args['--f'])
    framesize = int(args['--s'])
    nb_workers = int(args['--n'])
    overwrite = '--overwrite' in args
    format = args['--fmt']

    # Video datasets
    dataset = VideoDatasetWriter(path_in, path_out, framerate=framerate, size=(framesize, framesize),
                                 format=format, overwrite=overwrite)

    # Video dataloader
    loader = DataLoader(dataset, batch_size=1, sampler=None, shuffle=False, num_workers=nb_workers, pin_memory=False)

    # Write
    start = time.time()
    for k, batch in enumerate(loader):
        end = time.time()
        print("\rWriting video files... Progress = %.1f%% - ETA = %ds" % (
            100. * float(k + 1) / len(dataset), int((end - start) * (len(dataset) / float(k + 1) - 1.))), end=" ")