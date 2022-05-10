import numpy as np
import nibabel as nib
import argparse
import matplotlib.pyplot as plt

def display(filename):
    data = nib.load(filename).get_fdata()
    data_slice = data[:,:,100]
    plt.imshow(data_slice)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put the filename')
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()
    display(args.input)