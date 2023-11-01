import numpy as np

def zstandard(arr):
    _mu = np.nanmean(arr)
    _std = np.nanstd(arr)
    return (arr - _mu)/_std


satellite_database = {
    'Landsat 9': {
      'Folder': 'LC09/C02/T1_TOA',
      'Red': 'B4',
      'Green': 'B3',
      'Blue': 'B2',
      'NIR': 'B5',
      'SWIR1': 'B6',
      'SWIR2': 'B7',
      'Cloud': 'B11',
      'Shortname': 'L9',
    },
    'Landsat 8': {
      'Folder': 'LC08/C02/T1_TOA',
      'Red': 'B4',
      'Green': 'B3',
      'Blue': 'B2',
      'NIR': 'B5',
      'SWIR1': 'B6',
      'SWIR2': 'B7',
      'Cloud': 'B11',
      'Shortname': 'L8',
    },
    'Landsat 7': {
      'Folder': 'LE07/C02/T1_TOA',
      'Red': 'B3',
      'Green': 'B2',
      'Blue': 'B1',
      'NIR': 'B4',
      'SWIR1': 'B5',
      'SWIR2': 'B7',
      'Cloud': 'B6_VCID_2',
      'Shortname': 'L7',
    },
    'Landsat 5': {
      'Folder': 'LT05/C02/T1_TOA',
      'Red': 'B3',
      'Green': 'B2',
      'Blue': 'B1',
      'NIR': 'B4',
      'SWIR1': 'B5',
      'SWIR2': 'B7',
      'Cloud': 'B6',
      'Shortname': 'L5',
    },
    'Landsat 4': {
      'Folder': 'LT04',
      'Red': 'B3',
      'Green': 'B2',
      'Blue': 'B1',
      'NIR': 'B4',
      'SWIR1': 'B5',
      'SWIR2': 'B7',
      'Cloud': 'B6',
      'Shortname': 'L4',
    },
    'Sentinel-2': {
      'Folder': 'NA',
      'Red': 'B4',
      'Green': 'B3',
      'Blue': 'B2',
      'NIR': 'B8',
      'SWIR1': 'B11',
      'SWIR2': 'B12',
      'Cloud': 'QA60',
      'Shortname': 'S2',
    },
}
