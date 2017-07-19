from collections import namedtuple
from torch.utils.data import Dataset

CaptionedVideo = namedtuple('CaptionedVideo', ['video', 'caption'])


class VideoDataset(Dataset):

    def __len__(self):
        raise NotImplementedError('This should actually be implemented')

    def __getitem__(self):
        raise NotImplementedError('This should actually be implemented')
        return CaptionedVideo('video_tensor_of_size_CxTxWxH',
                              'encoded_caption_of_size_K')
