import glob
import os
from os import listdir
from os.path import isfile, join

import numpy as np


def get_path():
    """
    Args:
        None
    Returns:
        Returns the current path
    """
    return os.path.dirname(os.path.abspath(__file__))  # linux
    # TODO replace by:
    # from path import Path
    # Path(__file__).abspath().parent


def define_pkl_filename(particle, mechanics):
    """
    Args:
        params: class of imput parameters
    Returns:
        str: name of the output pkl file
    """
    outfile = mechanics.testcase
    outfile += "_r="
    outfile += (str)(np.round(particle.r_bar, 3))
    outfile += "_g0="
    outfile += (str)(mechanics.gamma_bar_0)
    outfile += "_gr="
    outfile += (str)(mechanics.gamma_bar_r)
    outfile += "_gfs="
    outfile += (str)(mechanics.gamma_bar_fs)
    outfile += "_gl="
    outfile += (str)(mechanics.gamma_bar_lambda)
    outfile += "_s0="
    outfile += (str)(mechanics.sigma_bar_0)
    outfile += "_sr="
    outfile += (str)(mechanics.sigma_bar_r)
    outfile += "_sfs="
    outfile += (str)(mechanics.sigma_bar_fs)
    outfile += "_sl="
    outfile += (str)(mechanics.sigma_bar_lambda)
    outfile += ".pkl"
    return outfile


def move_already_computed_files(testcase):
    path = get_path()
    directory = "already_computed_" + testcase
    dir_path = path + "/" + directory
    if not os.path.isdir(directory):
        os.mkdir(dir_path)
    file_list_in_path = [f for f in listdir(path) if isfile(join(path, f))]
    empty_pkl_files_created_counter = 0
    for i in file_list_in_path:
        if testcase in i[0:30]:
            file_size = os.path.getsize(path + "/" + i)
            if file_size > 1:  # check if the existing file is not empty
                os.rename(path + "/" + i, dir_path + "/" + i)
                # print('moved file ', i)
                with open(i, "w") as fp:
                    # print('created empty file named', i)
                    empty_pkl_files_created_counter += 1
                    pass
    # print('created %d empty .pkl files' %empty_pkl_files_created_counter)


def delete_empty_pkl_files():
    path = get_path()
    path_extension = "/*.pkl"
    file_path = path + path_extension
    deleted_file_counter = 0
    for filename in glob.glob(file_path):
        file_size = os.path.getsize(filename)
        if file_size < 1:
            os.remove(filename)
            deleted_file_counter += 1
    # print('deleted %d empty .pkl files' %deleted_file_counter)


def delete_pkl_files(testcase):
    path = get_path()
    path_extension = "/" + testcase + "*.pkl"
    # file_path = path / (testcase + "*.pkl")
    file_path = path + path_extension
    for filename in glob.glob(file_path):
        os.remove(filename)


def get_folder_size(path):
    """
    A function to compute the size of the current folder, in bytes

    Attributes:
        ----------
        path: str
            path of the folder

    Returns:
        -------
        total_size: float
            size of the current folder, in bytes
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size
