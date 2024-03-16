import utils.utils as utils

import numpy as np
import pandas as pd
import types

class f_cmean:
    def __init__(self,
                 data: pd.DataFrame,
                 n_cluster: int,
                 m : int,
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

        self.distance=distance
        self.arguments=arguments
        self.n_cluster=n_cluster
        self.m=m
        self.membership_matrix=None

    def membership(self,d:np.ndarray)-> np.ndarray:
        '''
        calculate the membership of each point to each cluster

        Args:
            d (np.ndarray): distance matrix

        Returns:
            np.ndarray: The membership matrix
        '''

        rows, cols = np.where(d == 0)
        if len(rows) > 0:
            d[rows, cols] = 1e-10
        for cluster in range(self.n_cluster):
            self.membership_matrix[cluster] = 1 / np.sum(
                (d[cluster] / (d)) ** (2 / (self.m - 1)),
                axis=0,
            )

    def cost_f(self,c:np.ndarray):
        '''
        Compute the cost function

        Args:
            U (np.ndarray): Membership function

        Returns:
            np.ndarray: _
        '''

        d= utils.matrix_distance(c,
                                 self.data,
                                 self.distance,
                                 self.arguments)

        cost = np.sum(np.sum(self.membership_matrix**self.m * d**2))


        return cost,d

    def update(self):
        '''
        Update center

        Args:
            U (np.ndarray): mempership matrix
        '''
        clusters = np.zeros((self.n_cluster, self.data.shape[1]))
        for i in range(clusters.shape[0]):
            clusters[i] = np.sum(self.membership_matrix[i]**self.m*self.data.T,axis=1)/(np.sum(self.membership_matrix[i]**self.m))

        return clusters

    def model(self,n_iter:int,tol):
        '''
        Model creation

        Args:
            U (np.ndarray): Data

        return: 
            Centroids
        '''

        p_loss=np.inf

        self.membership_matrix = np.random.rand(self.n_cluster,self.data.shape[0])

        for _ in range(n_iter):
            clusters = self.update()


            loss,distances = self.cost_f(clusters)


            if  np.abs(p_loss-loss) < tol :
                break

            p_loss = loss
            self.membership(distances)

        return clusters,self.membership_matrix
