import utils.utils as utils

import numpy as np
import pandas as pd
import types

class kmean:
    def __init__(self,
                 data: pd.DataFrame,
                 cluster: np.ndarray,
                 distance: types.FunctionType,
                 arguments: dict | None = None) -> None:
        '''
        __init__ create model

        Args:
            data (pd.DataFrame): data points
            cluster (np.ndarray): initial clusters
            distance (types.FunctionType): distance function tu use
            arguments (dict | None, optional): distance arguments
        '''

        self.data=np.array(data)
        self.clusters=cluster
        self.U=utils.matrix_distance(
            self.data,
            self.clusters,
            distance,
            arguments)
        