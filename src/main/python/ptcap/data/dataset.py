import pandas as pd
import gzip
import os

from collections import namedtuple
from torch.utils.data import Dataset

CaptionedVideo = namedtuple('CaptionedVideo', ['video', 'caption'])


class VideoDataset(Dataset):

    def __init__(self, path):
        self.annotations = self.open_annotations(path)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        video = self.open_video(index)
        raise NotImplementedError('This should actually be implemented')
        return CaptionedVideo('video_tensor_of_size_CxTxWxH',
                              'encoded_caption_of_size_K')

    def get_video_files(annotations, root):
        files = list(annotations.file)
        return [os.path.join(root, str(name)) for name in files]