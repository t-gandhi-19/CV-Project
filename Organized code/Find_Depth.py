import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm.notebook import tqdm
from conf import cfg, load_cfg_fom_args
import os



def main():
    print("Hello World!")
    load_cfg_fom_args(description="Victim_training")
    output_dir = cfg.SAVE_DIR
    output_dir_exists = os.path.exists(output_dir)
    if output_dir_exists:
        print("Output directory already exists!" , output_dir)
    else:
        os.makedirs(output_dir)
        print("Output directory created!", output_dir)

    # list of all folders in data directory
    data_dir = cfg.DATASET.DATA_DIR
    data_dir_exists = os.path.exists(data_dir)
    list_of_folders = []
    if data_dir_exists:
        print("Data directory exists!", data_dir)
        list_of_folders.extend(os.listdir(data_dir))

    else:
        print("Data directory does not exist!", data_dir)
   
    list_of_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print("List of folders in data directory:", list_of_folders)


if __name__ == "__main__":
    main()