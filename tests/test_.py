import pandas as pd, chemotaxis_toolbox as ct
from skimage import io
from pathlib import Path
import pytest

ipath = Path(ct.__file__[:-30], 'tests/')

test_in = pd.read_csv((ipath / 'test_data.csv'), index_col=0, na_filter=False)
uv_mask = io.imread((ipath / 'uv_mask.tiff'))
true1 = pd.read_csv((ipath / 'test1_out.csv'), index_col=0)
true4 = pd.read_csv((ipath / 'test4_out.csv'), index_col=0)
true1 = true1.round(8); true1 = true1.fillna(0)
true4 = true4.round(8); true4 = true4.fillna(0)

def simple_calcs(test_in, uv_mask):
    scale = 0.65
    test1 = ct.resolve_collisions(test_in, min_timepoints=19)
    test2 = ct.remove_slow_cells(test1, min_displacement=20, scaling_factor=scale)
    test3 = ct.remove_uv_cells(test2, uv_mask, min_timepoints=19, scaling_factor=scale)
    out1 = ct.get_chemotaxis_stats(test3, uv_mask, scaling_factor=scale)
    out1 = out1.round(8)
    out1 = out1.fillna(0)
    test_out = out1['Directed_velocity']
    return test_out

def calcs_by_interval(test_in, uv_mask):
    scale = 0.65
    test1 = ct.resolve_collisions(test_in, min_timepoints=19)
    test2 = ct.remove_slow_cells(test1, min_displacement=20, scaling_factor=scale)
    test3 = ct.remove_uv_cells(test2, uv_mask, min_timepoints=19, scaling_factor=scale)
    vel, ap, dir_vel = ct.get_chemotaxis_stats_by_interval(test3, uv_mask, scaling_factor=scale)
    dir_vel = dir_vel.round(8)
    dir_vel = dir_vel.fillna(0)
    return dir_vel

def test_simple_calcs():
    assert simple_calcs(test_in, uv_mask).equals(true1.Directed_velocity)

def test_calcs_by_interval():
    assert calcs_by_interval(test_in, uv_mask).equals(true4)
