import pickle

class StatCalculator(object):

    def __init__(self, annotations):
        """
        Calculate stats from all the expanded labels in all annotation files
        Args:
            annotations: list of paths to annotation files
        """

        self.stats = []

    def save_stats(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.caption_dict, f)
