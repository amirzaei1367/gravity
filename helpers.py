import torch
from settings import *
from sklearn.externals import joblib
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import numpy as np
from sklearn import svm
import pandas as pd
from numpy import linalg as LA
from sklearn.manifold import TSNE
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(fx, y):
    preds = fx.argmax(1, keepdim=True)
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def min_max(train_iterator):
    mean = 0.
    std = 0.
    min_total = np.array([])
    max_total = np.array([])

    for images, _ in train_iterator:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        min_total = np.append(min_total, images.min())
        max_total = np.append(max_total, images.max())
    #     if images.min() < min_total:
    #         min_total = images.min()
    #     if images.max() > max_total:
    #         max_total = images.max()

    min_total = min_total.min()
    max_total = max_total.max()
    print('min ', min_total)
    print('max ', max_total)

    return min_total, max_total

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        fx, _ = model(x)
        loss = criterion(fx, y)
        acc = calculate_accuracy(fx, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            fx, _ = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def criterion_kd(fx, fz, y, fzz, alfa=0.1):
    loss1 = nn.CrossEntropyLoss()(fx, y)
    loss2 = nn.KLDivLoss()(F.softmax(fz, dim=1),
                           F.softmax(fzz, dim=1))

    #     loss2 = nn.CosineSimilarity(dim=1)(fz, fzz)
    #     loss2=torch.mean(loss1, dim=0)
    #     loss2 = nn.KLDivLoss()(fz, fzz)
    return loss1 + alfa * loss2
    # return loss2

def train_kd(model, iterator, optimizer, criterion, device, centroid='modified_centroids', alfa=0.1):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
                                format(chose_dataset, chose_model, model_layer, centroid))
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        fx, fz = model(x)

        fzz = torch.tensor([centroid_list[lbl.item()] for lbl in y]).to(device)
        loss = criterion_kd(fx, fz, y, fzz, alfa)

        acc = calculate_accuracy(fx, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def extract_centroid(model, loader , device, ls_name, snapshot_name, centroid_name='original_centroids' ):
    '''
    :param model: the model
    :param loader: training or testing dataloader
    :param device: device to load the model
    :param snapshot_name: name of saved parameters of model
    :param ls_name: name of the latent space file
    :param centroid_name: name of the file that contains the centroid
    :return: centroid files
    '''
    ls_original = defaultdict(list)
    M_O = model
    M_O.load_state_dict(torch.load(log_path + "{}/{}/{}/{}.pth".
                                   format(chose_dataset, chose_model, model_layer, snapshot_name)))

    M_O.eval()
    for (x, y) in loader:
        x = x.to(device)
        y = y.to(device)
        fx, ls = M_O(x)
        ls_original[y.item()].append(ls.detach().squeeze().tolist())

    joblib.dump(ls_original, log_path + "{}/{}/{}/{}".
                format(chose_dataset, chose_model, model_layer, ls_name))

    centroid_list = defaultdict(list)
    for key, item in ls_original.items():
        kmeans = KMeans(init='k-means++', n_clusters=1, random_state=0).fit(item)
        centroid_list[key] = kmeans.cluster_centers_.tolist()[0]


    joblib.dump(centroid_list, log_path + "{}/{}/{}/{}".
                format(chose_dataset, chose_model, model_layer, centroid_name))
    return ls_original, centroid_list

#modify centroids based on my theory
####################################
def modified_centroid(centroid_path, ls_path, itr=4 , coef=1.5):
    '''
    :param centroid_path: indicates the name of original centroid file
    :param ls_path: indicates the name of the latent spaces file
    :param itr: shows the number of iteration needed for obtaining new centroid
    :return: the new centroids
    '''
    centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
                                format(chose_dataset, chose_model, model_layer, centroid_path))
    ls_original = joblib.load(log_path + "{}/{}/{}/{}".
                                format(chose_dataset, chose_model, model_layer, ls_path))
    for i in range(itr):
        for c_key, c_item in centroid_list.items():
            best_item = None
            th_distance = 0
            for item in ls_original[c_key]:
                np_mean = np.mean([np.linalg.norm(np.array(item) - np.array(b)) for b in list(centroid_list.values())])
                if np_mean > th_distance:
                    th_distance = np_mean
                    best_item = item.copy()
            centroid_list[c_key] = best_item

    joblib.dump(centroid_list, log_path + "{}/{}/{}/modified_centroids".
                format(chose_dataset, chose_model, model_layer))

    return centroid_list

    # db_distance = list(centroid_list.values())
    # lbl = list(centroid_list.keys())
    # reduced_data = PCA(n_components=2).fit_transform(db_distance)

    # fig, ax = plt.subplots()
    # ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
    #            marker='o', s=10, linewidths=3,
    #            color='r', zorder=10)
    #
    # for i, txt in enumerate(lbl):
    #     ax.annotate(txt, (reduced_data[:, 0][i], reduced_data[:, 1][i]))

#modify centroids based on normal vectors
# def modified_centroid(centroid_path, ls_path, itr=4 , coef=1.5):
#     '''
#     :param centroid_path: indicates the name of original centroid file
#     :param ls_path: indicates the name of the latent spaces file
#     :param itr: shows the number of iteration needed for obtaining new centroid
#     :return: the new centroids
#     '''
#     centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
#                                 format(chose_dataset, chose_model, model_layer, centroid_path))
#     ls_original = joblib.load(log_path + "{}/{}/{}/{}".
#                                 format(chose_dataset, chose_model, model_layer, ls_path))
#
#
#     bisectors = {}
#     for c_key_i, c_item_i in centroid_list.items():
#         bisector = 0
#         for c_key_j, c_item_j in centroid_list.items():
#             src = np.array(c_item_i)
#             dst = np.array(c_item_j)
#             bisector += ((src/LA.norm(src))  -  (dst/LA.norm(dst)))
#         bisectors[c_key_i] = bisector * coef
#
#     for c_key_i, c_item_i in centroid_list.items():
#         centroid_list[c_key_i] = list(np.array(centroid_list[c_key_i]) + bisectors[c_key_i])
#
#
#     joblib.dump(centroid_list, log_path + "{}/{}/{}/modified_centroids".
#                 format(chose_dataset, chose_model, model_layer))
#
#     return centroid_list
#
#     # db_distance = list(centroid_list.values())
#     # lbl = list(centroid_list.keys())
#     # reduced_data = PCA(n_components=2).fit_transform(db_distance)
#
#     # fig, ax = plt.subplots()
#     # ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
#     #            marker='o', s=10, linewidths=3,
#     #            color='r', zorder=10)
#     #
#     # for i, txt in enumerate(lbl):
#     #     ax.annotate(txt, (reduced_data[:, 0][i], reduced_data[:, 1][i]))

def distance_metric(latent_spaces):
    ## calculating the distance based on the SVM marginal distance of latent spaces
    svm_db = pd.DataFrame()
    for key, value in latent_spaces.items():
        tmp_db = pd.DataFrame(value)
        tmp_db['lbl'] = ([key] * len(value))
        svm_db = pd.concat([svm_db, tmp_db.copy()], axis=0)

    distance = 0
    for indx, (key, value) in enumerate(latent_spaces.items()):
        #     svm_dbb = pd.DataFrame()
        svm_dbb = svm_db.copy()
        svm_dbb.loc[svm_dbb.lbl != key, 'lbl'] = len(latent_spaces.keys())
        svm_dbb.loc[svm_dbb.lbl == key, 'lbl'] = len(latent_spaces.keys()) + 1

        y = svm_dbb['lbl']
        X = svm_dbb.drop(columns=['lbl'], axis=1)
        clf = svm.SVC(gamma='scale', kernel='linear', C=0.01)
        clf.fit(X, y)
        distance += 2 / np.linalg.norm(clf.coef_)
        print(indx)

    return distance

def distancetree_metric(centroid_path):
    ## calculating the distance of only centroid based on the spanning tree
    centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
                                format(chose_dataset, chose_model, model_layer, centroid_path))

    centroids = list(centroid_list.values())
    neigh = NearestNeighbors(n_neighbors=len(centroids))
    neigh.fit(centroids)
    A = neigh.kneighbors_graph(centroids, mode='distance')
    X = csr_matrix(A)
    Tcsr = minimum_spanning_tree(X)
    distance = Tcsr.toarray().sum()

    a = Tcsr.toarray()
    b = np.reshape(a, (-1,))
    b = np.where(b == 0, np.inf, b)
    minmum_dist = b.min()



    return minmum_dist, distance


def vis(ls_name, component=2, technique='tsne'):
    ls_original = joblib.load(log_path + "{}/{}/{}/{}".
                                format(chose_dataset, chose_model, model_layer, ls_name))
    # ls_original = joblib.load(ls_path)
    lbl = []
    color = []
    for index, (key, item) in enumerate(ls_original.items()):
        new_lbl = [key] * len(item)
        new_color = [key + 10] * len(item)
        lbl += new_lbl.copy()
        color += new_color
        if index == 0:
            out_arr = np.array(item)
        else:
            new_item = np.array(item)
            out_arr = np.vstack((out_arr, new_item))

    if technique == 'tsne':
        tsne_model = TSNE(n_components=component, random_state=SEED)
        tsne_data_ = tsne_model.fit_transform(out_arr)
        tsne_data = tsne_data_.copy()

        tsne_data = np.vstack((tsne_data.T, np.array(lbl), np.array(color))).T
        tsne_df = pd.DataFrame(data=tsne_data, columns=['dim_1', 'dim_2', 'lbl', 'color'])
        sns.FacetGrid(tsne_df, hue='color', size=6).map(plt.scatter, 'dim_1', 'dim_2', 'color').add_legend()
        plt.savefig(log_path + "{}/{}/{}/{}.png".
                                format(chose_dataset, chose_model, model_layer, ls_name))
        # plt.show()