import pickle
from datetime import datetime

import numpy as np
from numpy.core.multiarray import dot
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

from cnscraper import get_random_subset, split_features, convert_to_counts_df, load_simple_concepts_and_expand, \
    load_local_edgelist
from pmf import convert_to_coo_sparse_matrix
from vectorcomparer import find_vectors
if __name__ == '__main__':
    print("Loading edges")
    rss = load_local_edgelist(limit=100000)
    print("Splitting feats")
    conceptlist,featurelist,weightlist = split_features(rss)
    print("Converting to csm")
    conceptmap, featuremap, csm = convert_to_coo_sparse_matrix(conceptlist, featurelist, weightlist)
    csm = csm.tocsr()
    print("Normalizing")
    del conceptlist,featurelist,weightlist
    normed_matrix = normalize(csm, axis=1)

    X=normed_matrix
    print("Fitting")
    start_time = datetime.now()
    tsvd = TruncatedSVD(n_components=50,algorithm="arpack")
    # s = np.diagflat(s)
    # transformed = np.matmul(u,s)
    # transformed = np.matmul(transformed,v)
    tsvd.fit(X)
    end_time = datetime.now()
    print("Done")
    print(tsvd.singular_values_)
    skecth_scores = []
    nb_scores = []
    pairs = [("animal","dog"),("good","bad"),("motivation","inspiration"),("girl","chick"),("body","girl"),("britain","united_kingdom"),("warrior","war"),("car","table")]
    for pair in pairs:
        pref = "/c/en/"
        c1 = pair[0]
        c2 = pair[1]
        print("Comparing")
        print(c1)
        print(c2)
        truck_index = conceptmap[pref+c1]
        car_index = conceptmap[pref+c2]

        truck_row = X[truck_index].toarray()
        car_row = X[car_index].toarray()

        truck_low_dim = tsvd.transform(truck_row)[:,0]
        car_low_dim = tsvd.transform(car_row)[:,0]
        testdotres = dot(truck_low_dim,car_low_dim.transpose())
        skecth_scores.append(testdotres)
        print(testdotres)
        print("Comparing against Numberbatch")
        v1,v2 = find_vectors(c1,c2)
        nbdotres = dot(v1,v2)
        nb_scores.append(nbdotres)
        print(nbdotres)
    print(np.log(np.average(skecth_scores)) /np.log(np.average(nb_scores)))
    print(np.cov([skecth_scores,nb_scores]))
    print(str((end_time - start_time).seconds))