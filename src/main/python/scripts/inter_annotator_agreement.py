import os

import pandas as pd


LINES = 11
PATH = "/home/waseem/Downloads/Caption_Evaluation_"



def skip(lines):
    for i in range(lines):
        pass


def parse_file(folder_num, index):
    path = os.path.join(PATH + folder_num, str(index) + ".csv")
    if os.path.exists(path):
        csv_file = pd.read_csv(path)
        # Do stuff to get annotations

    return -1

if __name__ == '__main__':
    for i in range(10):
        parse_file("1", i + 1)
        parse_file("2", i + 1)
