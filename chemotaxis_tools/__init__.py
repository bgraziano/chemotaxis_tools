import pandas as pd, numpy as np
from chemotaxis_tools.support import *
pd.options.mode.chained_assignment = None # suppress waring messages for in-place dataframe edits

def resolve_collisions(df, min_timepoints):
    """
    Removes tracks where two cells collided, are segmented as one object, then later split
    apart. Collisions are identified by checking each unique 'id' to determine whether
    there are any idividual timepoints for which there are two objects associated with a
    single 'id' (i.e., two cells collided, split apart at a later time, but each retain
    the same object label). In such cases, data for all timepoints on and after the
    splitting event (following the initial collision) are removed for that 'id'. 'id'
    values are assigned in the 'validate_dataframe' function.

    Parameters
    ----------
    df: DataFrame
        Must include columns labeled 'Time', 'Experiment_number', 'Cell_line',
        'Cell_number', 'x', and 'y'. 'Time', 'Experiment_number', and 'Cell_number', must
        be series of integers; 'Cell_line' must be a series of strings; 'x' and 'y' must
        be series of floats.

    min_timepoints: integer
        After rows are removed based on the criteria described above, the remaining data
        for a given 'cell_id' are removed if there are fewer remaining timepoints than the
        value set for 'min_timepoints'

    Returns
    -------
    output: DataFrame
        This DataFrame contains all the original columns. If not already present, a new
        'id' column containing a unique string for each unique cell track will be added.
        Rows are deleted where collisions occurred.

    """
    df = validate_dataframe(df)
    assert min_timepoints >= 3, 'The value of the "min_timepoints" must be 3 or greater!'
    assert min_timepoints < len(df['Time'].unique()) - 3, 'The value of "min_timepoints" is set too high!'
    id_list = df['id'].unique()
    for item in id_list:
        if len(df[df['id'] == item]['Time'].unique()) < len(df[df['id'] == item]):
            sub_df = df[df['id'] == item]
            cutoff_time = sub_df.index[sub_df.Time.duplicated()][0]
            item_index = sub_df.index[sub_df['id'] == item]
            drop_index = item_index[item_index >= cutoff_time]
            df.drop(index=drop_index, inplace=True)
    for item in df['id'].unique():
        sub_table = df[df['id'] == item]
        if len(sub_table) <= min_timepoints:
            df = df[df['id'] != item]
    df.reset_index(drop=True, inplace=True)
    return df

def remove_slow_cells(df, min_displacement, scaling_factor): # Removes cells with max displacement is below set value. Useful for removing dead cells or debris
    """
    Removes tracks for cells whose maximum displacements are less than the specified
    value. This is useful for removing dead cells, debris, etc. Maximum displacement is
    assessed by comparing the distance between a cell's intial x-y position with its x-y
    position at every other timepoint. For example, if a cell travels in a complete circle
    (its final x-y position is equal to its initial x-y displacement) it would have a
    maximun displacement greater than zero and equal to the diameter of its circular path.
    Cells are assigned unique 'id' values in the 'validate_dataframe' function.

    Parameters
    ----------
    df: DataFrame
        Must include columns labeled 'Time', 'Experiment_number', 'Cell_line',
        'Cell_number', 'x', and 'y'. 'Time', 'Experiment_number', and 'Cell_number', must
        be series of integers; 'Cell_line' must be a series of strings; 'x' and 'y' must
        be series of floats. IMPORTANT: Ensure that 'x' and 'y' are in units of pixels.

    min_displacement: float
        Cell tracks with a value less than 'min_displacement' will be removed from the
        DataFrame. IMPORTANT: This value should be real units of length (nanometers,
        microns, etc. and NOT pixels).

    scaling_factor: float
        Factor for conversion of 'x' and 'y' series of 'df' from pixels to real units of
        length. IMPORTANT: The real units of length that the 'x' and 'y' series are being
        converted to should match the units of the 'min_displacement' parameter.

    Returns
    -------
    output: DataFrame
        This DataFrame contains all the original columns. If not already present, a new
        'id' column containing a unique string for each unique cell track will be added.
        Cell tracks (rows) are removed in cases where the minimun displacement criteria
        are not satisfied. All data for a unique cell 'id' are removed.

    """
    df = validate_dataframe(df)
    assert type(float(min_displacement)) is float, '"min_displacement" must be an integer!'
    assert type(float(scaling_factor)) is float, '"scaling_factor" must be a float!'
    time_step = int(df['Time'].diff().mode())
    frame_num = np.amax(df['Time']) // time_step + 1
    cell_list = df.id.unique()
    df_out = pd.DataFrame(columns=[])
    for item in cell_list:
        current_cell = df[df['id'] == item]
        init_x = current_cell['x'].iloc[0]
        init_y = current_cell['y'].iloc[0]
        current_cell['Displacement'] = ((current_cell['x'] - init_x)**2 + (current_cell['y'] - init_y)**2)**0.5 * scaling_factor
        if current_cell.Displacement.max() >= min_displacement:
            df_out = df_out.append(current_cell.iloc[:frame_num], ignore_index=True, sort=False)
    df_out.drop(columns=['Displacement'], inplace=True)
    return df_out

def remove_uv_cells(df, uv_img, min_timepoints, scaling_factor):
    """
    Removes portions of tracks where cells come into contact with UV light. Furthermore,
    if a cell enters and then exits the area of light projection, the portion of the
    track where the cell exits the light projection area is also removed. This results
    in all data being removed for a cell track as soon at that cell touches the
    boundary of the light projection area.

    Parameters
    ----------
    df: DataFrame
        Must include columns labeled 'Time', 'Experiment_number', 'Cell_line',
        'Cell_number', 'x', and 'y'. 'Time', 'Experiment_number', and 'Cell_number', must
        be series of integers; 'Cell_line' must be a series of strings; 'x' and 'y' must
        be series of floats. IMPORTANT: Ensure that 'x' and 'y' are in units of pixels.

    uv_img: ndarray
        A 2d binary image (x-y) containing the mask of the UV light projected by the DMD.
        The mask of the UV light should be a single continuous object, and there should
        be no other objects aside from the UV mask.

    min_timepoints: integer
        After rows are removed based on the criteria described above, the remaining data
        for a given 'cell_id' are removed if there are fewer remaining timepoints than the
        value set for 'min_timepoints'.

    scaling_factor: float
        Factor for conversion of 'x' and 'y' series of 'df' from pixels to real units of
        length. IMPORTANT: The real units of length that the 'x' and 'y' series are being
        converted to should match the units of the 'min_displacement' parameter.

    Returns
    -------
    output: DataFrame
        This DataFrame contains all the original columns. If not already present, a new
        'id' column containing a unique string for each unique cell track will be added.
        Cell tracks (rows) are removed in cases where the minimun displacement criteria
        are not satisfied. All data for a unique cell 'id' are removed.

    """
    assert min_timepoints >= 3, 'The value of the "min_timepoints" must be 3 or greater!'
    assert type(float(scaling_factor)) is float, '"scaling_factor" must be a float!'
    df = validate_dataframe(df)
    uv_stats = get_uv_pos(uv_img, scaling_factor)
    time_step = int(df['Time'].diff().mode())
    try:
        len(df['Relative_time'])
    except:
        df = get_relative_time(df)

    # For a given track, remove all timepoints where cell is inside the UV circle
    df['Distance_to_target'] =  ((df['x'] - uv_stats[0])**2 + (df['y'] - uv_stats[1])**2)**0.5 * scaling_factor
    df = df[df['Distance_to_target'] > uv_stats[2]]

    # Removes all cells whose starting positions were already inside the UV circle
    for item in df['id'].unique():
        sub_table = df[df['id'] == item]
        if len(sub_table[sub_table['Relative_time'] == 0]) == 0:
            df = df[df['id'] != item]
    df.reset_index(drop=True, inplace=True)

    # After above steps, there may be broken tracks if a cell travelled into the UV circle but then left again at a later time.
    # This 'while' loop removes all timepoints at these later times such that every track is continuous and always terminates
    # at the timepoint where the cell first comes into contact with the UV circle.
    score_max = time_step
    while score_max > 0:
        offset_table = df.copy()
        offset_table.drop(index=0, inplace=True)
        offset_table.reset_index(drop=True, inplace=True)
        df['score'] = offset_table['Relative_time'] - df['Relative_time'] - time_step
        score_max = df['score'].max()
        drop_rows = df[df['score'] > 0].index.values + 1
        df.drop(index=drop_rows, inplace=True)
        df.reset_index(drop=True, inplace=True)
    df.drop(columns=['score'], inplace=True)

    # after trimming back cell tracks that enter the UV circle, this further removes cells not present for a user-set minimum number of time points.
    for item in df['id'].unique():
        sub_table = df[df['id'] == item]
        if len(sub_table) <= min_timepoints:
            df = df[df['id'] != item]
    return df

def get_chemotaxis_stats(df, uv_img, scaling_factor):
    """
    Calculates the 'Angular_persistence', 'Velocity', and 'Directed_velocity' for each
    timepoint of each unique cell. The 'Angular_persistence' and 'Directed_velocity'
    metrics are equivalent to the 'angular bias' and 'directed speed' described in
    https://doi.org/10.15252/msb.20156027. 'Angular_persistence' values range from -1 to
    1, with postive values indicating movement towards the region of UV photo-uncaging,
    negative values indicating movement away (i.e., chemorepulsion), and '0' indicating
    random movement. 'Directed_velocity' is calculated from multiplying the 'Velocity'
    and 'Angular_persistence' of a cell for each timepoint.

    Parameters
    ----------
    df: DataFrame
        Must include columns labeled 'Time', 'Experiment_number', 'Cell_line',
        'Cell_number', 'x', and 'y'. 'Time', 'Experiment_number', and 'Cell_number', must
        be series of integers; 'Cell_line' must be a series of strings; 'x' and 'y' must
        be series of floats. IMPORTANT: Ensure that 'x' and 'y' are in units of pixels.

    uv_img: ndarray
        A 2d binary image (x-y) containing the mask of the UV light projected by the DMD.
        The mask of the UV light should be a single continuous object, and there should
        be no other objects aside from the UV mask.

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
    df = validate_dataframe(df)
    uv_stats = get_uv_pos(uv_img, scaling_factor)
    time_step = int(df['Time'].diff().mode()) # for calculating velocity, etc. later on
    try:
        len(df['Relative_time'])
    except:
        df = get_relative_time(df)
    df['x_from_center'] = uv_stats[0] - df['x']
    df['y_from_center'] = uv_stats[1] - df['y']
    df.reset_index(drop=True, inplace=True)
    df_out = get_ap_vel(df, time_step, scaling_factor)
    df_out.loc[df['Relative_time'] == 0, ['Angular_persistence', 'Velocity', 'Directed_velocity']] = np.NaN # Clear data since velocity, etc. can't be calculated for first timepoint.
    df_out.reset_index(drop=True, inplace=True)
    return df_out

def get_chemotaxis_stats_by_interval(df, uv_img, scaling_factor):
    """
    Similar to the "get_chemotaxis_stats" function, calculates 'Angular_persistence',
    'Velocity', and 'Directed_velocity'. However, for a given cell, these values are
    calculated across multiple time intervals i, where i ranges from 1 to the maximum
    number of time intervals divided by 2. For each i, the mean value is determined for
    each metric. The overall approach is similar in spirit to how mean-squared
    displacement is calculated for a particle undergoing diffusion.

    Parameters
    ----------
    df: DataFrame
        Must include columns labeled 'Time', 'Experiment_number', 'Cell_line',
        'Cell_number', 'x', and 'y'. 'Time', 'Experiment_number', and 'Cell_number', must
        be series of integers; 'Cell_line' must be a series of strings; 'x' and 'y' must
        be series of floats. IMPORTANT: Ensure that 'x' and 'y' are in units of pixels.

    uv_img: ndarray
        A 2d binary image (x-y) containing the mask of the UV light projected by the DMD.
        The mask of the UV light should be a single continuous object, and there should
        be no other objects aside from the UV mask.

    scaling_factor: float
        Typically supplied by the calling function. Factor for conversion of 'x' and 'y'
        series of 'df' from pixels to real units of length. IMPORTANT: If designing a
        pipeline with other functions in this toolbox, ensure that the same real units of
        length are used in all cases (e.g., everything is coverted to microns).

    Returns
    -------
    output: Three DataFrames
        DataFrames summarize the mean 'Velocity', 'Angular_persistence', or
        'Directed_velocity', respectively, for each cell over each time interval. The
        numerical column headings indicate the duration of each time interval, using the
        same units as the 'Time' column of the input DataFrame.

    """
    df = validate_dataframe(df)
    assert len(df[df['Time'] == 0]) > 0, ''
    ap_collection = pd.DataFrame(columns=[])
    vel_collection = pd.DataFrame(columns=[])
    uv_stats = get_uv_pos(uv_img, scaling_factor) # Gets the size and location of UV light circle. This is used later for removing cells that enter this area
    original_time_step = int(df['Time'].diff().mode())
    try:
        len(df['Relative_time'])
    except:
        df = get_relative_time(df)
    df['x_from_center'] = uv_stats[0] - df['x']
    df['y_from_center'] = uv_stats[1] - df['y']
    cell_list = df['id'].unique()

    for pos, item in enumerate(cell_list):
        mean_ap_list = []; mean_vel_list = []
        ap_table = df[df['id'] == item]
        ap_table.reset_index(drop=True, inplace=True)
        total_time_intervals = np.amax(ap_table['Relative_time'].values) // original_time_step // 2
        assert len(ap_table) == len(ap_table['Relative_time'].unique()), 'Duplicate entries present. Resolve collisions before running this function.'
        assert total_time_intervals == (len(ap_table) - 1) // 2, 'Check time intervals!'

        for interval in np.arange(total_time_intervals + 1)[-total_time_intervals:]:
            index_list = []
            for x in np.arange(interval): # generate list of indices to pull from 'ap_table' to satisfy the current interval (e.g., for interval=2, get every other row)
                index_list = index_list + np.arange(x, len(ap_table), interval).tolist()
            sub_table = ap_table.iloc[index_list]
            time_step = interval * original_time_step
            sub_table = get_ap_vel(sub_table, time_step, scaling_factor)
            sub_table.drop(index=np.arange(interval), inplace=True)
            mean_ap_list.append(sub_table['Angular_persistence'].mean())
            mean_vel_list.append(sub_table['Velocity'].mean())

        ap_collection = ap_collection.append(pd.DataFrame(data=mean_ap_list).transpose())
        vel_collection = vel_collection.append(pd.DataFrame(data=mean_vel_list).transpose())
        print('Cell ' + str(pos + 1) + ' of ' + str(len(cell_list)) + ' done.', end='\r')

    ap_collection.columns = (np.arange(len(ap_collection.columns)) + 1) * original_time_step
    ap_collection.columns = ap_collection.columns.astype(str)
    vel_collection.columns = (np.arange(len(vel_collection.columns)) + 1) * original_time_step
    vel_collection.columns = vel_collection.columns.astype(str)
    dir_vel_collection = pd.DataFrame(columns=[])
    for col in ap_collection.columns:
        dir_vel_collection[col] = ap_collection[col] * vel_collection[col]

    ap_collection['Cell_num'] = cell_list; vel_collection['Cell_num'] = cell_list; dir_vel_collection['Cell_num'] = cell_list
    ap_collection.set_index('Cell_num', inplace=True); vel_collection.set_index('Cell_num', inplace=True); dir_vel_collection.set_index('Cell_num', inplace=True)
    return vel_collection, ap_collection, dir_vel_collection

def plot_tracks(df, uv_img):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.patches import Circle
    df = validate_dataframe(df)
    uv_x, uv_y, uv_radius = get_uv_pos(uv_img, 1)
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 14
    rcParams['lines.linewidth'] = 1
    plot_list = df['Cell_line'].unique()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13, 6))
    for pos, item in enumerate(plot_list):
        plot_table = df[['Cell_line', 'Time', 'id', 'x', 'y']]
        plot_table = plot_table[plot_table['Cell_line'] == item]
        plot_table_x = plot_table.pivot(index='Time', columns='id', values='x')
        plot_table_y = plot_table.pivot(index='Time', columns='id', values='y')
        if pos == 0:
            ax1.set_title(item + ' cell tracks')
            ax1.set(xlabel='X', ylabel='Y')
            ax1.plot(plot_table_x, plot_table_y)
            uv_circle = Circle((uv_x, uv_y),uv_radius, alpha=0.3, color='purple')
            ax1.add_patch(uv_circle)
        if pos == 1:
            ax2.set_title(item + ' cell tracks')
            ax2.set(xlabel='X')
            ax2.plot(plot_table_x, plot_table_y)
            uv_circle = Circle((uv_x, uv_y),uv_radius, alpha=0.3, color='purple')
            ax2.add_patch(uv_circle)
        if pos > 1:
            print('Only two cell lines can be plotted at once. Skipping the remaining lines.')
            continue
    plt.show()
    return
