"""Training script.
Usage:
  train.py <config_path>
  train.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""

from docopt import docopt

from ptcap.data.config_parser import YamlConfig
from ptcap.ptcap import caption

if __name__ == '__main__':
    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args['<config_path>'])

    # Run captioning model
    caption(config_obj)
