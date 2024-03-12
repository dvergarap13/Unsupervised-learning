import numpy as np
import pandas as pd
from typing import Callable

from utils.distances import distance
import utils.utils as utils
import types



class mountain:
    def __init__(self,
                 data: pd.DataFrame,
                 l: int,
                 distance: types.FunctionType,
                 arguments: dict | None = None,
                 sigma: float = 1,
                 beta: float = 1):
        '''
        __init__ Create de model

        Args:
            data (pd.DataFrame | np.ndarray): Data to train the model
            l (int): number of partitions of the grid
            distance (Callable): the distance function
            sigma (float): the sigma parameter
            beta (float): the beta parameter
        '''

        self.data = np.array(data)
        self.V = utils.create_grid(l,self.data.shape[1])
        self.distances_data = utils.matrix_distance(self.V,
                                               self.data,
                                               distance,
                                               arguments)

        self.distances_grid = utils.matrix_distance(
            self.V,
            self.V,
            distance,
            arguments)
        
        self.sigma=sigma
        self.beta=beta

    def mountain_function(self)->np.array:
        '''
        mountain_function calculate the mountain height  for a vector of distance X

        Args:
            X (np.ndarray): Vector of distance between data and V


        Returns:
            np.array: _description_
        '''
        
        m_height = np.sum(np.exp(-self.distances_data**2/(2*self.sigma**2)),axis=1)

        return m_height

    def destruction(self,m:np.ndarray,index:int)->np.array:
        '''
        destruction calculate the new potential clusters

        Args:
            v (np.array): Potential clusters
            c1 (np.array): Cluster
        Returns:
            np.array: _description_
        '''
        m_c=m[index]

        m_new= m - m_c*np.exp( - self.distances_grid[index]**2 / (2*self.beta**2) )

        return m_new

    def model(self,n_iter:int):
        '''
        model Run the model

        Args:
            n_iter (int): _description_
        '''

        centers_index=set()

        mountain=self.mountain_function()

        center=np.argmax(mountain)

        

        for _ in range(n_iter):

            if center in centers_index:
                break


            mountain = self.destruction(mountain,center)
            
            centers_index.add(center)

            center=np.argmax(mountain)

        centers_index.add(center)

        centers=self.V[np.array(list(centers_index))]
        return centers_index,centers,mountain 

    def plot(self,centers:np.ndarray,name:str):
        '''
        plot first 3 clusters

        Args:
            centers (np.ndarray): clusters
            name (str): name of the graph
        '''

        return
            

        
        