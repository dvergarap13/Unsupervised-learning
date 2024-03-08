import numpy as np
from typing import Callable


def distance_lp(x: np.ndarray ,
                y:np.ndarray,
                p:int)-> float :
    '''
    distance_lp computes the distance between 2 points

    Args:
        x (np.ndarray): first vector
        y (np.ndarray): second vector
        p (int): the p value

    Returns:
        float: the distance between x and y
    '''
    assert x.shape==y.shape

    if p==0:
        distancia=np.sum(np.max(x,y))
    else:
        distancia=np.sum(np.abs(x-y)**p)**(1/p)

    return distancia

def mahalanobis(x: np.ndarray ,y:np.ndarray,cov : np.ndarray )->float:
    '''
    mahalanobis computes the distance between 2 points

    Args:
        x (np.ndarray): first vector
        y (np.ndarray): second vector
        cov (np.ndarray): the p value

    Returns:
        float: the distance between x and y
    '''

    assert x.shape==y.shape

    if not x.shape[0] ==  cov.shape[0]:
        x = x.T
        y = y.T

    inv = np.linalg.pinv(cov)

    return np.sqrt(
        np.dot(np.dot((x-y).T,inv),(x-y))
    )


def cosine_distance(x: np.ndarray ,y:np.ndarray)-> float :
    '''
    cosine_distance computes the distance between 2 points

    Args:
        x (np.ndarray): first vector
        y (np.ndarray): second vector
        

    Returns:
        float: the distance between x and y
    '''
    assert x.shape==y.shape

    x_norm=np.linalg.norm(x)
    y_norm=np.linalg.norm(y)

    if x_norm==y_norm:
        similarity=0
    else:
        similarity=np.dot(x,y)/(x_norm*y_norm)
        

    return 1 - similarity

def distance(dist: str)->Callable:
    '''
    distance return the distance

    Args:
        dist (str): the distance

    Returns:
        callable: distance function
    '''

    if dist=="lp":
        return distance_lp
    elif dist == "mahalanobis":
        return mahalanobis
    elif dist == "cosine":
        return cosine_distance
    else:
        raise ValueError("Invalid distance")