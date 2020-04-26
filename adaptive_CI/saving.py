"""
This script contains helping functions for better saving format. 
"""

import numpy as np
import pandas as pd
import os
import subprocess
from time import time
from os.path import dirname, realpath, join, exists
from os import makedirs, chmod
from getpass import getuser

__all__ = [
    "compose_filename",
]


def compose_filename(prefix, extension):
    """
    Creates a unique filename based on Github commit id and time.
    Useful when running in parallel on server.

    INPUT:
        - prefix: file name prefix
        - extension: file extension

    OUTPUT:
        - fname: unique filename
    """
    # Tries to find a commit hash
    try:
        commit = subprocess\
            .check_output(['git', 'rev-parse', '--short', 'HEAD'],
                          stderr=subprocess.DEVNULL)\
            .strip()\
            .decode('ascii')
    except subprocess.CalledProcessError:
        commit = ''

    # Other unique identifiers
    rnd = str(int(time() * 1e8 % 1e8))
    sid = tid = jid = ''
    ident = filter(None, [prefix, commit, jid, sid, tid, rnd])
    basename = "_".join(ident)
    fname = f"{basename}.{extension}"
    return fname
