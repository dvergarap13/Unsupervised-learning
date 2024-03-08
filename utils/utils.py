import numpy as np
import pandas as pd
from utils.distances import distance
from typing import Callable
from sklearn import preprocessing
import json

def create_grid(l: int, N: int)->np.ndarray:
    '''
    create_grid Create a grid of dimension  (NxN) with l elements on each axis fro 0 to 1.

    Args:
        partition (int): number of partitions
        dimension (int): number of dimension

    Returns:
        np.ndarray: the grid
    '''

    interval = np.linspace(0,1,l)
    grid = np.tile(interval,N).reshape(-1,l)

    return np.array(np.meshgrid(*grid)).T.reshape(-1,N)

def matrix_distance(X:np.ndarray,
                    Y: np.ndarray,
                    metric: Callable,
                    arguments:dict=None)->np.ndarray:
    '''
    matrix_distance Pairwise  distance between rows in two matrices x and y using the provided metric function.

    Args:
        X (np.ndarray): Matrix  X
        Y (np.ndarray): Matrix Y
        metric (Callable): distance function
        p  (int): power for the metric
        argument (dictionary): additional parameters for the distance

    Returns:
        np.ndarray: matrix of distances
    '''
    assert X.shape[1] == Y.shape[1]

    if  arguments is None:
        arguments={}
    elif metric.__name__=="mahalanobis":
        arguments["cov"] = np.cov(X.T)

    n=X.shape[0]
    m=Y.shape[0]

    distances= np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            distances[i,j]=metric(x=X[i],y=Y[j],**arguments)

    return distances


def read_data(name:str)->pd.DataFrame:
    '''
    read_data read data

    Args:
        name (str): name of the  dataset to load

    Returns:
        pd.DataFrame: data
    '''

    path= "/data/"+name

    
    if name.endswith(".csv") or name.endswith(".txt"):
        reader = pd.read_csv(path, sep = None, iterator = True)
        inferred_sep = reader._engine.data.dialect.delimiter
        data=pd.read_csv(path,sep=inferred_sep)
    if name.endswith(".xslx") or name.endswith(".xls"):
        data=pd.read_excel(path)
    if name.endswith(".json"):
        data=pd.read_json(path)
    else:
        raise ValueError("Type not supported")

    return data

def clean_data(data:pd.DataFrame)->pd.DataFrame:
    '''
    clean_data clean data

    Args:
        data (pd.DataFrame): raw data

    Returns:
        pd.DataFrame: clean data
    '''

    data = data.dropna()


    data = pd.DataFrame(preprocessing.minmax_scale(data,axis=1),columns=data.columns)


    return data



def get_params()->pd.DataFrame:
    '''
    Clean the data by removing missing values and scaling the data.

    Parameters:
    data (pd.DataFrame): raw data

    Returns:
    pd.DataFrame: clean data
    '''
    with open('parameters.json') as json_file:
        parameters = json.load(json_file)

    distances=[]
    for i in range(parameters["distances"]):
        distances.append(distance(i))

    return (
        parameters.get("name"),
        distances,
        parameters["arguments"],
        parameters["kmeans"],
        parameters["fuzzy_kmeans"],
        parameters["mountain_clustering"].get("sigma"),
        parameters["mountain_clustering"].get("beta"),
        parameters["mountain_clustering"].get("l"),
        parameters["substractive_clustering"]
        
    )


    