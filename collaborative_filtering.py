import numpy as np
import pandas as pd
import math
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def cal_cosine_similarity(data_items):
    """Calculate the cosine similarity for a sparse
    matrix. Return a new dataframe matrix with similarities.
    """
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse.transpose())
    cos_sim_matrix = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
    return cos_sim_matrix



def pearson_similarity(data_items):
    """Calculate the adjusted cosine similarity.
     Return a new dataframe matrix with similarities.
    """
    matrix_a = data_items.mean(axis=1)
    matrix_a =matrix_a.to_numpy()
    item_mean_subtracted = data_items - matrix_a[:, None]
    adj_sim_matrix = 1 - squareform(pdist(item_mean_subtracted.T, 'cosine'))
    return adj_sim_matrix



def jaccard_similarity(ratings_nan):
    """Calculate the jaccard similarity with hamming metric.
        Return a new dataframe matrix with similarities.
    """
    # true=nan , false = a number for rate
    train_set_bool = pd.isna(ratings_nan)
    train_set_bool *= 1
    train_set_bool = np.logical_not(train_set_bool).astype(int)
    jac_sim = 1 - pairwise_distances(train_set_bool, metric="hamming")
    # optionally convert it to a DataFrame
    jac_sim = pd.DataFrame(jac_sim, index=train_set_bool.T.columns, columns=train_set_bool.columns)
    return jac_sim



def dice_similarity(ratings_nan):
    """Calculate the dice similarity for a sparse
          matrix. Return a new dataframe matrix with similarities.
      """
    train_set_bool = pd.isna(ratings_nan)
    train_set_bool *= 1
    train_set_bool = np.logical_not(train_set_bool).astype(int)
    dicesim= np.zeros((ratings_nan.index.size, ratings_nan.columns.size))
    for i in range(ratings_nan.index.size):
        for j in range(ratings_nan.columns.size):
            intersection = np.logical_and(train_set_bool.to_numpy()[i], train_set_bool.to_numpy()[j])
            dicesim[i][j]= 2. * intersection.sum()/ (train_set_bool.to_numpy()[i].sum() + train_set_bool.to_numpy()[j].sum())
    return dicesim



def neighbors_prediction(ratings_nan, ratings_nan_weighted, similarity, k):
    """
    Find the k-nearest neighbors and
    calculate the prediction value with mean average and weighted mean

    """
    k= int(k)
    #find the neighbors
    for i in range (ratings_nan.index.size):
        for j in range(ratings_nan.columns.size):
            if (pd.isna(ratings_nan.values[i][j])):
                ss=( ratings_nan.index[i],ratings_nan.columns[j])
                neighbors = similarity.values[j]
                temp = np.copy(neighbors)
                neighbors.sort()
                neighbors= neighbors[len(neighbors)-k-1:-1]
                neighbors_results= [0 for n in range (k)]

                # k nearest neighbors
                for n in range (len(neighbors)):
                    neighbors_results[n] = np.where((neighbors[n] == temp))

                l=-1
                for m in range (len(neighbors_results)):
                    if (len(neighbors_results[m][0])>1):
                        l+=1
                        if(l < len(neighbors_results[m][0])):
                            neighbors_results[m] =np.array(tuple([neighbors_results[m][0][l]]))
                        else:
                            l=0
                            neighbors_results[m] = np.array(tuple([neighbors_results[m][0][l]]))
                    else:
                        l=-1


                #calculate the weighted arithmetic mean and the mean average
                sum=0
                mean_weighted=0
                mean_average=0
                numOfsim=0
                for n in range (len(neighbors_results)):
                    if(~np.isnan(ratings_nan.values[i][neighbors_results[n]])):
                        mean_weighted += (ratings_nan.values[i][neighbors_results[n]] * neighbors[n])
                        mean_average += ratings_nan.values[i][neighbors_results[n]]
                        sum+=neighbors[n]
                        numOfsim+=1

                #if the k is a low number and the ratings from the k-nearest
                if mean_weighted==0:
                    mean_weighted=np.array([1])
                    mean_average=np.array([1])
                    numOfsim=1
                    sum=1

                #weighted average
                ratings_nan_weighted.values[i][j] =mean_weighted[0]/sum
                #mean average
                ratings_nan.values[i][j] = mean_average[0]/numOfsim





def predictSim(ratings, ratings_nan, items, users,k, error_rates_mean_average,error_rates_weighted_average):

    # Calculate similarities item-item
    jacsim = jaccard_similarity(ratings_nan)

    dicesim = dice_similarity(ratings_nan)
    dicesim = pd.DataFrame(dicesim, columns= items , index=items)

    cossim= cal_cosine_similarity(ratings_nan.apply(lambda row: row.fillna(row.mean()), axis=1))

    pearsim = pearson_similarity(ratings_nan.apply(lambda row: row.fillna(row.mean()), axis=1))
    pearsim = pd.DataFrame(pearsim, columns= items , index=items)

    #copy ratings
    test_pearsim  = ratings_nan.copy()
    test_cossim = ratings_nan.copy()
    test_dicesim  = ratings_nan.copy()
    test_jacsim  = ratings_nan.copy()
    test_pearsim_weighted = ratings_nan.copy()
    test_cossim_weighted = ratings_nan.copy()
    test_dicesim_weighted = ratings_nan.copy()
    test_jacsim_weighted = ratings_nan.copy()

    #calculate the predicted value
    neighbors_prediction(test_jacsim, test_jacsim_weighted,  jacsim, k)
    neighbors_prediction(test_pearsim, test_pearsim_weighted, pearsim, k)
    neighbors_prediction(test_dicesim , test_dicesim_weighted,dicesim, k)
    neighbors_prediction(test_cossim,test_cossim_weighted, cossim, k)

    #calculate mean squared error for mean average

    pearson_error = mean_squared_error(test_pearsim,ratings)
    cosine_error= mean_squared_error(test_cossim,ratings)
    dice_error = mean_squared_error(test_dicesim, ratings)
    jac_error = mean_squared_error(test_jacsim, ratings)

    # calculate mean squared error for mean average
    pearson_error_weighted = mean_squared_error(test_pearsim_weighted, ratings)
    cosine_error_weighted = mean_squared_error(test_cossim_weighted, ratings)
    dice_error_weighted = mean_squared_error(test_dicesim_weighted, ratings)
    jac_error_weighted = mean_squared_error(test_jacsim_weighted, ratings)

    #sum the errors for each iteration

    error_rates_mean_average['pearson_error'] += pearson_error
    error_rates_mean_average['cosine_error'] += cosine_error
    error_rates_mean_average['dice_error'] += dice_error
    error_rates_mean_average['jac_error'] += jac_error

    error_rates_weighted_average['pearson_error'] +=  pearson_error_weighted
    error_rates_weighted_average['cosine_error'] += cosine_error_weighted
    error_rates_weighted_average['dice_error'] += dice_error_weighted
    error_rates_weighted_average['jac_error'] += jac_error_weighted

    return  error_rates_mean_average ,error_rates_weighted_average




def dataset_creation():
    num_uuids = 100
    num_of_items = 100

    #*create Ids for users*
    uuids = pd.Series(['userid' + str(i)for i in range(1, num_uuids + 1)])

    #* create Ids for items*
    items = pd.Series(['itemid' + str(i) for i in range(1, num_of_items + 1)])

    #* A dataframe for users&items, create by random uniform (1, 10)
    df = pd.DataFrame(np.random.uniform(1.0, 10.0, size=(100, 100)), index=list(uuids), columns=list(items) )
    print(df.head())

    #* save df table, items table and uuids table  *
    df.to_csv("ratings.csv", index = False)
    items.to_csv("items.csv", index = False)
    uuids.to_csv("users.csv", index = False)



def recommendation_system():
    """
        Call the "dataset_creation()" to create the dataset.
        If the csv files exist, then read them.
        Call the predictSim, T times,  computing the similarities item-item for predicting annotation
    """


    # Create the dataset
    #dataset_creation()


    # Read the dataset
    items = pd.read_csv("items.csv")
    users = pd.read_csv("users.csv")
    ratings = pd.read_csv("ratings.csv")
    ratings.index = users

    T=10
    k = input("Please give a number for the k-nearest neighbors: ")

    error_pear = error_cos = error_dice = error_jac = error_pear_wt = error_cos_wt = error_dice_wt = error_jac_wt = 0

    error_rates_mean_average = {
        'pearson_error': error_pear,
        'cosine_error': error_cos,
        'dice_error': error_dice,
        'jac_error': error_jac

    }
    error_rates_weighted_average = {
        'pearson_error': error_pear_wt,
        'cosine_error': error_cos_wt,
        'dice_error': error_dice_wt,
        'jac_error': error_jac_wt
    }

    for i in range(T):

        # put 20%  random nan values
        ratings_nan = ratings.mask(np.random.random(ratings.shape) < .2)
        #predicting and finding mean squared eroor
        error_rates_mean_average, error_rates_weighted_average = predictSim(ratings,ratings_nan,items,users,k,error_rates_mean_average,error_rates_weighted_average)


    # calculate the mean average, for each error in T iterations
    error_rates_mean_average['pearson_error'] = error_rates_mean_average['pearson_error'] /T
    error_rates_mean_average['cosine_error'] = error_rates_mean_average['cosine_error'] / T
    error_rates_mean_average['dice_error'] = error_rates_mean_average['dice_error'] / T
    error_rates_mean_average['jac_error'] = error_rates_mean_average['jac_error'] / T

    error_rates_weighted_average['pearson_error'] = error_rates_weighted_average['pearson_error'] / T
    error_rates_weighted_average['cosine_error'] = error_rates_weighted_average['cosine_error'] / T
    error_rates_weighted_average['dice_error'] = error_rates_weighted_average['dice_error'] / T
    error_rates_weighted_average['jac_error'] = error_rates_weighted_average['jac_error'] / T

    print(error_rates_mean_average)
    print(error_rates_weighted_average)

    # plot bar error for mean average
    plt.bar(range(len(error_rates_mean_average)), list(error_rates_mean_average.values()), align='center')
    plt.xticks(range(len(error_rates_mean_average)), list(error_rates_mean_average.keys()))
    plt.yscale('log', base=10)
    plt.show()

    #plot bar error for weighted mean
    plt.bar(range(len(error_rates_weighted_average)), list(error_rates_weighted_average.values()), align='center',
            color=('r'))
    plt.xticks(range(len(error_rates_weighted_average)), list(error_rates_weighted_average.keys()))
    plt.yscale('log', base=10)
    plt.show()


if __name__ == '__main__':
    recommendation_system()

