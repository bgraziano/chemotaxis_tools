import pandas as pd, numpy as np
import datetime, os
from pathlib import Path

def validate_dataframe(df):
    """
    Verifies that input dataframe is formatted properly and creates a unique cell 'id' for
    each cell track. Also creates a text file for logging analysis steps if it is the first
    time this function has been called on a given dataframe.

    Parameters
    ----------
    df: DataFrame
        Typically provided by the calling function.

    Returns
    -------
    output: DataFrame
        This DataFrame contains all the original columns. If not already present, a new
        'id' column containing a unique string for each unique cell track will be added.
        Rows are deleted where collisions occurred.

    """
    assert type(df) is pd.DataFrame, 'Values must be formatted as a Pandas DataFrame!'
    assert type(df['Cell_line'][0]) is str, '"Cell_line" column entries must be strings!'
    assert type(int(df['Experiment_number'][0])) is int, '"Experiment_number" column entries must be integers!'
    df['Experiment_number'] = df['Experiment_number'].astype(int)
    assert type(int(df['Cell_number'][0])) is int, '"Cell_number" column entries must be integers!'
    assert type(int(df['Time'][0])) is int, '"Time" column entries must be integers!'
    assert type(df['x'][0]) is np.float64, '"x" column entries must be floats!'
    assert type(df['y'][0]) is np.float64, '"y" column entries must be floats!'
    df.sort_values(by=['Cell_line', 'Experiment_number', 'Cell_number'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    try:
        len(df['id'])
    except:
        df['id'] = df['Cell_line'] + '_' + df['Experiment_number'].astype(str) + '_' + df['Cell_number'].astype(str) # assigns a unique 'cell_id' to each cell track

        # Initializes log file for recording analysis steps of subsequent functions.
        homedir = os.path.expanduser('~')
        write_location = Path(homedir, 'chemotaxis logs/')
        if not os.path.exists(write_location):
            os.mkdir(write_location)
            print('New directory for log files created at: "' + str(write_location) + '"')
        time = datetime.datetime.now().strftime('%H_%M_%S')
        time_text = datetime.datetime.now().strftime('%H:%M:%S')
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        log_file_name = 'chemotaxis analysis log ' + date + ' ' + time + '.txt'
        try:
            logfile = open(write_location / log_file_name, 'x')
            print('New log file "' + log_file_name + '" created at "' + str(write_location) + '"')
        except:
            logfile = open(write_location / log_file_name, 'a')
        logfile.write('Analysis was performed on ' + date + ' at ' + time_text + ' on cell lines ' + str(df['Cell_line'].unique()) + ':\n\n')
        logfile.close()
    return df

def get_uv_pos(uv_img, scaling_factor):
    """
    Calculates the center (x-y coordinates) and radius of the UV light projection mask.

    Parameters
    ----------
    uv_img: ndarray
        Typically provided by the calling function. A 2d binary image (x-y) containing
        the mask of the UV light projected by the DMD. The mask of the UV light should be
        a single continuous object, and there should be no other objects aside from the
        UV mask.

    scaling_factor: float
        Typically provided by the calling function. Factor for conversion of 'x' and 'y'
        series of 'df' from pixels to real units of length. IMPORTANT: If designing a
        pipeline with other functions in this toolbox, ensure that the same real units of
        length are used in all cases (e.g., everything is converted to microns).

    Returns
    -------
    x: float; y: float
        The row (x) and column (y) coordinates specifying the center of the UV light
        mask. 'x' and 'y' are in units of pixels.

    radius: float
        The radius of the UV mask identified in "UV-img". This value is in real units of
        length, as specified by the "scaling_factor" parameter.

    """
    from math import sqrt, pi
    from skimage import measure
    assert type(uv_img) is np.ndarray, '"UV_img" must be a numpy array!'
    assert len(uv_img.shape) == 2, '"UV_img" must be a single x-y plane!'
    assert len(np.unique(uv_img)) == 2, '"UV_img" must be binary!'
    uv_img_labeled = measure.label(uv_img, connectivity=2)
    area = [r.area for r in measure.regionprops(uv_img_labeled)]
    center = [list(r.weighted_centroid) for r in measure.regionprops(uv_img_labeled, intensity_image=uv_img)]
    assert len(area) == len(center) == 1, '"UV_img" must contain only one object (i.e., the UV light source)'
    center = np.asarray(center); y = center[0][0]; x = center[0][1]
    y = uv_img.shape[1] - y # need to 'flip' this value since rows are labeled top-to-bottom
    radius = sqrt(area[0] / pi) * scaling_factor
    return x, y, radius

def get_relative_time(df):
    """
    Calculates the 'Relative_time' for each time point of a unique cell track (as
    determined by its 'id', which is assigned in the "validate_dataframe" function.
    For the initial timepoint where a cell with a unique 'id' appears, it is assigned a
    time of '0', regardless of the value specified in the 'Time' column. All subsequent
    timepoints are then assigned increasing values using the same interval as the 'Time'
    column.

    Parameters
    ----------
    df: DataFrame
        Typically provided by the calling function.

    Returns
    -------
    output: DataFrame
        This DataFrame contains all the original columns with the further addition of a
        'Relative_time' column.

    """
    df = validate_dataframe(df)
    id_list = df['id'].unique()
    df['Relative_time'] = df['Time']
    for item in id_list:
        sub_table = df[df['id'] == item]
        sub_table['Relative_time'] = sub_table['Relative_time'] - np.amin(sub_table['Relative_time'])
        df.loc[df['id'] == item, 'Relative_time'] = sub_table['Relative_time']
    return df

def get_ap_vel(df, time_step, scaling_factor): # Calculates 'angular persistence', 'velocity', and 'directed velocity'
    """
    Primary function called by "get_chemotaxis_stats" and
    "get_chemotaxis_stats_by_interval". Calculates the 'Angular_persistence', 'Velocity',
    and 'Directed_velocity' for each timepoint of each unique cell.

    Parameters
    ----------
    df: DataFrame
        Typically supplied by the calling function. Must include columns labeled 'Time',
        'Experiment_number', 'Cell_line', 'Cell_number', 'x', and 'y'. 'Time',
        'Experiment_number', and 'Cell_number', must be series of integers; 'Cell_line'
        must be a series of strings; 'x' and 'y' must be series of floats. IMPORTANT:
        Ensure that 'x' and 'y' are in units of pixels.

    time_step: integer
        Typically supplied by the calling function. This value specifies the duration of
        the interval between each timepoint for a cell track.

    scaling_factor: float
        Typically supplied by the calling function. Factor for conversion of 'x' and 'y'
        series of 'df' from pixels to real units of length. IMPORTANT: If designing a
        pipeline with other functions in this toolbox, ensure that the same real units of
        length are used in all cases (e.g., everything is coverted to microns).

    Returns
    -------
    output: DataFrame
        This DataFrame contains all the original columns with the further addition of
        'Velocity', 'Angular_persistence', and 'Directed_velocity' columns.

    """
    diff_df = df[['x', 'y', 'x_from_center', 'y_from_center']].diff()
    dot_product = df['x_from_center'] * diff_df['x_from_center'] + df['y_from_center'] * diff_df['y_from_center']
    magnitude = (df['x_from_center']**2 + df['y_from_center']**2)**0.5 * (diff_df['x_from_center']**2 + diff_df['y_from_center']**2)**0.5
    df['Angular_persistence'] = dot_product / magnitude * -1
    df['Velocity'] = (diff_df['x']**2 + diff_df['y']**2)**0.5 * scaling_factor / time_step
    df['Directed_velocity'] = df['Velocity'] * df['Angular_persistence']
    return df

def get_dir_ac(df, time_step, scaling_factor):
    """
    """
    diff_df = df[['x', 'y']].diff()
    shift_df = df[['x','y']].shift(-1)
    diff_shift_df = diff_df[['x', 'y']].shift(-1)
    dot_product_1 = df['x'] * diff_df['x'] + df['y'] * diff_df['y']
    magnitude_1 = (df['x']**2 + df['y']**2)**0.5 * (diff_df['x']**2 + diff_df['y']**2)**0.5
    cosines_1 = dot_product_1 / magnitude_1 * -1
    dot_product_2 = shift_df['x'] * diff_shift_df['x'] + shift_df['y'] * diff_shift_df['y']
    magnitude_2 = (shift_df['x']**2 + shift_df['y']**2)**0.5 * (diff_shift_df['x']**2 + diff_shift_df['y']**2)**0.5
    cosines_2 = dot_product_2 / magnitude_2 * -1
    df['ap1'] = cosines_1
    df['ap2'] = cosines_2
    df['ap_differences'] = cosines_2 - cosines_1

    return df

def update_log(summary_string):
    homedir = os.path.expanduser('~')
    write_location = Path(homedir, 'chemotaxis logs')
    file_list = []
    for root, dirs, files in os.walk(write_location, topdown=False): # Generate a list of all files in the log directory
        for name in files:
            file_list.append(os.path.join(root, name))
    log_file_name = file_list[-1] # Get the most recent file that was written
    logfile = open(write_location / log_file_name, 'a')
    time = datetime.datetime.now().strftime('%H:%M:%S')
    logfile.write(time + '   ' + summary_string + '\n\n')
    logfile.close()
    return
