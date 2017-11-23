import numpy as np

import cv2
import glob
from PIL import Image

from torch.utils.data import Dataset
from gulpio import GulpDirectory


class VideoDataset(Dataset):
    def __init__(self, annotation_parser, tokenizer, preprocess=None):
        self.tokenizer = tokenizer
        self.video_paths = annotation_parser.get_video_paths()
        self.video_ids = annotation_parser.get_video_ids()
        self.captions = annotation_parser.get_captions()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        """
        Return a Tuple like
        ("video_tensor_of_size_CxTxWxH", "encoded_caption_of_size_K")
        """

        video = self._try_get_video(index)
        if self.preprocess is not None:
            video = self.preprocess(video)
        tokenized_caption = self._get_tokenized_caption(index)
        return video, self.captions[index], np.array(tokenized_caption)

    def _get_video(self, index):
        pass

    def _get_tokenized_caption(self, index):
        return self.tokenizer.encode_caption(self.captions[index])

    def _try_get_video(self, index):
        try:
            return self._get_video(index)
        except:
            print("\nSkipping", self.video_paths[index], "...")
            return self._try_get_video(index + 1)


class JpegVideoDataset(VideoDataset):
    """
    Open bursted frames saved in JPG files. This class does on-the-fly resizing
    using PIL and requires the size of frames ([128, 128] e.g.)
    as one extra-parameter.
    """

    def __init__(self, size=[128, 128], resample=Image.BICUBIC, *args,
                 **kwargs):
        super(JpegVideoDataset, self).__init__(*args, **kwargs)
        self.size = size
        self.resample = resample

    def _get_video(self, index):
        dirname = self.video_paths[index]
        frames = [np.array(Image.open(path).convert("RGB").
                           resize(self.size, resample=self.resample))
                  for path in glob.glob(dirname + "/*.jpg")]
        return np.array(frames)


class NumpyVideoDataset(VideoDataset):
    """
    Load video saved in a NPZ file.
    """

    def _get_video(self, index):
        dirname = self.video_paths[index]
        path = glob.glob(dirname + "/*.npz")[0]
        video = np.load(path)["arr_0"]
        return video


class GulpVideoDataset(VideoDataset):

    def __init__(self,  annotation_parser, tokenizer, gulp_dir, preprocess=None,
                 size=None):
        super().__init__(annotation_parser, tokenizer, preprocess=preprocess)

        # instantiate the GulpDirectory
        self.gulp_dir = GulpDirectory(gulp_dir)
        self.size = tuple(size) if size else None

    def _get_video(self, index):

        frames, _ = self.gulp_dir[self.video_ids[index]]
        if self.size:
            frames = [cv2.resize(f, self.size) for f in frames]
        return np.array([np.array(f) for f in frames])
