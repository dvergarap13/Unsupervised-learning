from Unsupervised.mountain_clustering import mountain
from Unsupervised.substracting_clustering import substracting
from Unsupervised.kmeans import kmean
from Unsupervised.fuzzy_cmeans import f_cmean

from utils.utils import get_params,read_data,clean_data




def main():
    (
        name,
        distances,
        arguments,
        kmeans_params,
        fuzzy_kmeans_params,
        mountain_clustering_sigma,
        mountain_clustering_beta,
        mountain_clustering_l,
        substractive_clustering_r_a,
        substractive_clustering_r_b)=get_params()

    data = read_data("Iris.csv")

    data = clean_data(data)

    modelo=f_cmean(data,4,2,distances[0],arguments[0])

    centroids = modelo.model(100,0.0001)
    print(1)
    


if  __name__ == "__main__":
    main()