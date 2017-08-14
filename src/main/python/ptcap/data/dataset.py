import numpy as np

import glob
from PIL import Image

from torch.utils.data import Dataset


class VideoDataset(Dataset): 

    def __init__(self, annotation_parser, tokenizer, preprocess=None):
        self.tokenizer = tokenizer
        self.video_paths = annotation_parser.get_video_paths()
        self.captions = annotation_parser.get_captions()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        """
        Return a Tuple like
        ('video_tensor_of_size_CxTxWxH', 'encoded_caption_of_size_K')
        """

        video = self._get_video(index)
        if self.preprocess is not None:
            video = self.preprocess(video)
        tokenized_caption = self._get_tokenized_caption(index)
        return video, self.captions[index], np.array(tokenized_caption)

    def _get_video(self, index):
        pass

    def _get_tokenized_caption(self, index):
        return self.tokenizer.encode_caption(self.captions[index])


class JpegVideoDataset(VideoDataset):
    """
    Open bursted frames saved in JPG files. This class does on-the-fly resizing
    using PIL and requires the size of frames ([128, 128] e.g.)
    as one extra-parameter.
    """

    def __init__(self, size=[128,128], resample=Image.BICUBIC, *args, **kwargs):
        super(JpegVideoDataset, self).__init__(*args, **kwargs)
        self.size = size
        self.resample = resample

    def _get_video(self, index):
        dirname = self.video_paths[index]
        frames = [np.array(Image.open(path).convert('RGB').
                           resize(self.size, resample=self.resample))
                  for path in glob.glob(dirname + "/*.jpg")]
        return np.array(frames)


class NumpyVideoDataset(VideoDataset):
    """
    Load video saved in a NPZ file.
    """

    def __init__(self, *args, **kwargs):
        super(NumpyVideoDataset, self).__init__(*args, **kwargs)

    def _get_video(self, index):
        dirname = self.video_paths[index]
        path = glob.glob(dirname + "/*.npz")[0]
        temp = np.load(path)["arr_0"]
        return temp

        # dirname = self.video_paths[index]
        # frames = [np.array(np.load(path))
        #           for path in glob.glob(dirname + "/*.npz")]
        # return np.array(frames)
