import numpy as np
from ptcap.data.annotation_parser import AnnotationParser
import glob

from collections import namedtuple
from torch.utils.data import Dataset

CaptionedVideo = namedtuple('CaptionedVideo', ['video', 'caption'])


class VideoDataset(Dataset):

    def __init__(self, annotation, tokenizer_obj):
        self.annotation = annotation
        self.tokenizer = tokenizer_obj
        self.video_paths = list(annotation[AnnotationParser.FILE_FIELD])
        self.captions = list(annotation[AnnotationParser.CAPTION_FIELD])



    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        """
        Return a Tuple like
        ('video_tensor_of_size_CxTxWxH', 'encoded_caption_of_size_K')
        """

        video = self._get_video(index)
        tokenized_caption = self._get_tokenized_caption(index)
        return CaptionedVideo(video, tokenized_caption)

    def _get_video(self, index):
        """
           Open burst frames saved in JPG files. This class does on-the-fly
           resizing using PIL and requires the size of frames ([128, 128] e.g.)
            as one extra-parameter.
        """

        video_path = self.video_paths[index]
        return []

    def _get_tokenized_caption(self, index):
        return self.tokenizer.encode_caption(self.captions[index])

    def _get_original_caption(self, index):
        return self.tokenizer.tokenize(self.captions[index])
