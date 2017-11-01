import os

import pandas as pd


LINES = 11
PATH = "/home/waseem/Downloads/Caption_Evaluation_"



def skip(lines):
    for i in range(lines):
        pass


if __name__ == '__main__':
    for i in range(10):
        path = os.path.join(PATH + "1", str(i + 1) + ".csv")
        csv_file1 = pd.read_csv(path)
        # Do stuff

        path = os.path.join(PATH + "2", str(i + 1) + ".csv")
        csv_file2 = pd.read_csv(path)
