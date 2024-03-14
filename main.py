from Unsupervised.mountain_clustering import mountain
from Unsupervised.substracting_clustering import substracting

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

    modelo = substracting(data=data,
                          distance=distances[0],
                          arguments=arguments[0],
                          r_b=substractive_clustering_r_a[3],
                          r_a=substractive_clustering_r_b[3])

    corrida = modelo.model(100)


g    print(1)
    


if  __name__ == "__main__":
    main()