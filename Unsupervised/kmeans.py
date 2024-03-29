import utils.utils as utils

import numpy as np
import pandas as pd
import types

class kmean:
    def __init__(self,
                 data: pd.DataFrame,
                 cluster: np.ndarray,
                 n_cluster: int,
                 random:bool,
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
        if random:
            self.clusters= self.data[np.random.choice(self.data.shape[0],n_cluster,replace=False)]
        else:
            self.clusters=cluster

        self.distance=distance
        self.arguments=arguments
        self.distances=utils.matrix_distance(
            self.clusters,
            self.data,
            self.distance,
            self.arguments)

    def membership(self,d:np.ndarray)-> np.ndarray:
        '''
        calculate the membership of each point to each cluster

        Args:
            d (np.ndarray): distance matrix

        Returns:
            np.ndarray: The membership matrix
        '''

        index=np.argmin(d,axis=0)

        U=np.zeros((self.clusters.shape[0],self.data.shape[0]))

        cols=np.arange(self.data.shape[0])

        U[index,cols]=1

        return U

    def cost_f(self,U:np.ndarray,d:np.ndarray):
        '''
        Compute the cost function

        Args:
            U (np.ndarray): Membership function

        Returns:
            np.ndarray: _
        '''


        return np.sum(U*d**2)

    def update(self,U:np.ndarray):
        '''
        Update center

        Args:
            U (np.ndarray): mempership matrix
        '''

        for i in range(self.clusters.shape[0]):
            self.clusters[i] = 1/np.sum(U[i])*(np.sum(U[i]*self.data.T,axis=1))

    def model(self,n_iter:int,tol):
        '''
        Model creation

        Args:
            U (np.ndarray): Data

        return: 
            Centroids
        '''

        p_loss=np.inf

        for _ in range(n_iter):


            membership = self.membership(self.distances)

            self.update(membership)

            loss = self.cost_f(membership,self.distances)

            if  np.abs(p_loss-loss) < tol :
                break

            p_loss = loss

        labels = np.argmax(membership,axis=0)

        return labels,self.clusters

    

        

        

          
        

        
        
        