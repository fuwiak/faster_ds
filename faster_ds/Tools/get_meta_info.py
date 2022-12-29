import numpy as np
import pandas as pd


class GetInfo:
    
    @staticmethod
    def get_info(df):
        raise NotImplementedError

    @staticmethod
    def num_megabytes(df):
        return sum(df.memory_usage()/1024**2)

