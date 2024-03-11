from Unsupervised.mountain_clustering import mountain

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
        substractive_clustering)=get_params()

    data = read_data("Iris.csv")

    data = clean_data(data)

    modelo = mountain(data=data,
                      l=mountain_clustering_l,
                      distance=distances[0],
                      arguments=arguments[0],
                      sigma=mountain_clustering_sigma[0],
                      beta=mountain_clustering_beta[0])

    corrida = modelo.model(100)

    print(1)
    


if  __name__ == "__main__":
    main()