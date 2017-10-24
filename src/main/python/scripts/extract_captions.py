"""Caption extraction script.
Usage:
  extract_captions.py <log_file> <parsed_output>
  extract_captions (-h | --help)

Options:
  <log_file>            Path to the log file to be parsed.
  <parsed_output>       Output path of the parsed file.
  -h --help             Show this screen.
"""

import csv
import re

from docopt import docopt


PATH = "/media/waseem/5CF8-A6B2/out2.txt"

OUT_PATH = "/home/waseem/20bn-gitrepo/pytorch-captioning/clean_out2.csv"


def parse_line(line):
    line = re.sub("'|]|\n","", line)
    caption, string = line.split("[")
    string_list = string.split(", ")
    processed_string = ""
    for token in string_list:
        if token != "<END>":
            processed_string += token + " "
        else:
            break

    return processed_string


def parse_file(input_path, output_path):
    read_file = open(input_path)
    with open(output_path, "w") as write_file:
        write_csv_file = csv.writer(write_file)
        write_csv_file.writerow(["TARGET", "PREDICTION"])
        target = ""
        prediction = ""
        write_to_csv = False
        line = read_file.readline()
        while line != "":
            if re.match("__TARGET__", line):
                target = parse_line(line)
            elif re.match("PREDICTION", line):
                prediction = parse_line(line)
                write_to_csv = True

            if write_to_csv:
                write_csv_file.writerow([target, prediction])
                write_to_csv = False

            line = read_file.readline()

    read_file.close()


if __name__ == '__main__':
    # Get argument
    args = docopt(__doc__)

    parse_file(*args)

    parse_file(PATH, OUT_PATH)