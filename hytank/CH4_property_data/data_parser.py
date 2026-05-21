"""
@File    :   data_parser.py
@Description : Read methane (LNG) property data from the data files.

Mirrors the public API of ``hytank.H2_property_data.data_parser`` so
that ``MethaneProperties`` can build its surrogates with the same
two-call pattern (``get_sat_property`` / ``get_property``) as
``HydrogenProperties``.
"""

import os

import numpy as np


def get_sat_property(name):
    """Get saturated methane property from data file for a range of temperatures.

    Parameters
    ----------
    name : str
        Column name desired from data file

    Returns
    -------
    numpy array
        Numpy array with data from requested column
    """
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saturated_properties.txt")

    with open(file, "r") as f:
        for line in f:
            columns = line.split("\t")
            break

    if name not in columns and name + "\n" not in columns:
        raise ValueError(f"{name} is an invalid column name")
    try:
        idx = columns.index(name)
    except ValueError:
        idx = columns.index(name + "\n")

    data = np.genfromtxt(file, delimiter="\t", skip_header=1)

    return data[:, idx]


def get_property(name, phase="both", pressure=None):
    """Get property of methane from data files for a range of temperatures and pressures.

    Parameters
    ----------
    name : str
        Column name desired from data file
    phase : str
        Select whether to return liquid data, vapor data, or both by setting this to
        "liquid", "vapor", or "both", by default both
    pressure : str, optional
        Desired pressure in bar formatted as a string to match the data file name, by default None
        where it will return data from all files

    Returns
    -------
    numpy array
        Flattened 1D array with data from all data files or just the specified one if pressure argument defined
    """
    if phase not in ["liquid", "vapor", "both"]:
        raise ValueError(f'Phase input must be either "liquid", "vapor", or "both", not "{phase}"')

    data_dir = os.path.dirname(os.path.abspath(__file__))
    dir_files = os.listdir(data_dir)
    if pressure is None:
        data_files = []
        for file in dir_files:
            if "_bar_properties.txt" in file:
                data_files.append(file)
        data_files.sort()
    else:
        file = pressure + "_bar_properties.txt"
        if file not in dir_files:
            raise ValueError(f"No data file with a pressure of {pressure} found")
        data_files = [file]

    data = np.array([], dtype=float)

    for filename in data_files:
        file = os.path.join(data_dir, filename)
        with open(file, "r") as f:
            idx_first_vapor = 0
            for i, line in enumerate(f):
                if i == 0:
                    columns = line.split("\t")
                    if phase == "both":
                        break
                    continue
                is_liquid = "liquid" in line.split("\t")[-1]
                if is_liquid:
                    idx_first_vapor = i
                else:
                    break

        if name not in columns and name + "\n" not in columns:
            raise ValueError(f"{name} is an invalid column name")
        try:
            idx = columns.index(name)
        except ValueError:
            idx = columns.index(name + "\n")

        data_cur_file = np.genfromtxt(file, delimiter="\t", skip_header=1)

        row_start = 0
        row_end = data_cur_file.shape[0]
        if phase == "liquid":
            row_end = idx_first_vapor
        elif phase == "vapor":
            row_start = idx_first_vapor

        data = np.hstack((data, data_cur_file[row_start:row_end, idx]))

    return data
