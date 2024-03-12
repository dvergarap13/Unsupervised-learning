import numpy as np
import pandas as pd
from typing import Callable

from utils.distances import distance
import utils.utils as utils
import types



class substracting:
    def __init__(self,
                 data: pd.DataFrame,
                 distance: types.FunctionType,
                 arguments: dict | None = None,
                 r_b: float = 1,
                 r_a: float = 1):
        '''
        __init__ Create de model

        Args:
            data (pd.DataFrame | np.ndarray): Data to train the modeñ
            distance (Callable): the distance function
            r_b (float): the sigma parameter

        '''

        self.data = np.array(data)
        self.distances_data = utils.matrix_distance(self.data,
                                               self.data,
                                               distance,
                                               arguments)
        
        self.r_a=r_a
        self.r_b = r_b
        
    def density_function(self)->np.array:
        '''
        mountain_function calculate the mountain height  for a vector of distance X

        Args:
            X (np.ndarray): Vector of distance between data and data


        Returns:
            np.array: _description_
        '''
        
        density = np.sum(np.exp(-self.distances_data**2/((self.r_a/2)**2)),axis=1)

        return density

    def destruction(self,density:np.ndarray,index:int)->np.array:
        '''
        destruction calculate the new potential clusters

        Args:
            v (np.array): Potential clusters
            c1 (np.array): Cluster
        Returns:
            np.array: _description_
        '''
        d_c=density[index]

        density_new=density - d_c*np.exp( - self.distances_data[index]**2 / ((self.r_b/2)**2) )

        return density_new

    def model(self,n_iter:int):
        '''
        model Run the model

        Args:
            n_iter (int): _description_
        '''

        centers_index=set()

        candidates_d=self.density_function()

        center=np.argmax(candidates_d)

        

        for _ in range(n_iter):

            if center in centers_index:
                break


            candidates_d = self.destruction(candidates_d,center)
            
            centers_index.add(center)

            center=np.argmax(candidates_d)

        centers_index.add(center)

        centers=self.data[np.array(list(centers_index))]
        return centers_index,centers,candidates_d 

    def plot(self,centers:np.ndarray,name:str):
        '''
        plot first 3 clusters

        Args:
            centers (np.ndarray): clusters
            name (str): name of the graph
        '''

        return
            

        
        