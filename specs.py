# coding: utf-8

"""
Constant values and specifications.
Note that some of these values are configured via environment variables.
"""

import os


# extract environment variables with hardcoded defaults
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
cernbox_data_url = "https://cernbox.cern.ch/index.php/s/6CK0CwO5W6HSgZB"

# composite variables
cernbox_dl_pattern = cernbox_data_url.rstrip("/") + "/download?path={}&files={}"

# constants
n_constituents = 200
n_events_per_file = 50000
n_files = {
    "train": 20,
    "valid": 8,
    "test": 8,
}
