import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-filepath', help='Path to the numpy file')
args = parser.parse_args()

data = np.load(args.filepath)
print(data)