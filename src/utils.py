import pandas as pd
import numpy as np

import os
import sys
import dill

from src.exception import Custon_exception
from src.logger import logging

def save_path(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise Custon_exception(e, sys)