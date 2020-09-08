import torch
from settings import *
import cw #carlini attack
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
import os
import logging

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition

def set_logger():
    if not os.path.exists(log_path + "{}".format(chose_dataset)):
        os.mkdir(log_path + "{}".format(chose_dataset))

    if not os.path.exists(log_path + "{}/{}".format(chose_dataset, chose_model)):
        os.mkdir(log_path + "{}/{}".format(chose_dataset, chose_model))

    if not os.path.exists(log_path + "{}/{}/{}".format(chose_dataset, chose_model, model_layer)):
        os.mkdir(log_path + "{}/{}/{}".format(chose_dataset, chose_model, model_layer))

    logging.basicConfig(filename=log_path + "{}/{}/{}/log.log".format(chose_dataset, chose_model, model_layer),
                        format='%(asctime)s: %(funcName)s(): %(lineno)d:\t %(message)s',
                        filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger
logger = set_logger()

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

def conv_matrix(model, iterator, device, chose_dataset, chose_model, model_layer,  alfa, index, black_box=False ):
    model.eval()

    gt = np.array([])
    pred = np.array([])
    with torch.no_grad():
        for (x, y) in iterator:
            np.append(gt, y.numpy())

            x = x.to(device)
            # y = y.to(device)

            if black_box == True:
                fx = model(x)
            else:
                fx, _, _ = model(x)

            pd = fx.argmax(1, keepdim=True)
            np.append(pred, pd.numpy())


    conf_mtx = confusion_matrix(gt, pred)
    with open(log_path + "{}/{}/{}/{}/{}.mtx".format(chose_dataset, chose_model, model_layer, alfa, index ),
              "w") as p:
        json.dump(conf_mtx, p)

def evaluate(model, iterator, criterion, device, black_box=False):
    gt = np.array([])
    pred = np.array([])
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            gt = np.append(gt, y.numpy())
            x = x.to(device)
            y = y.to(device)

            if black_box == True:
                fx = model(x)
            else:
                fx, _, _ = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)
            pd = fx.argmax(1, keepdim=True)
            pred = np.append(pred, pd.cpu().numpy())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        # acc = np.sum( pred == gt)
    return epoch_loss / len(iterator), epoch_acc / len(iterator), pred, gt
    # return epoch_loss / len(iterator), acc, pred, gt


def criterion_kd(fx, fz, y, fzz, l_1, l_11, alfa=0.1):
    # loss_ce = nn.CrossEntropyLoss()(fx, y)
    # loss_kl = nn.KLDivLoss()(F.softmax(fz, dim=1),
    #                        F.softmax(fzz, dim=1))
    if alfa > entropy_threshold_at:
        loss_ce = torch.tensor(0.0, dtype=float)
        # print(f"fz type {type(fz)} with shape {fz.shape }" )
        # print(f"fzz type {type(fzz)} with shape {fzz.shape}" )
        loss_mse_1 = nn.MSELoss()(fz, fzz)
        loss_mse_i = nn.MSELoss()(l_1, l_11)
        gravity_loss = loss_mse_1+loss_mse_i
    else:
        loss_ce = nn.CrossEntropyLoss()(fx, y)
        loss_mse_1 = nn.MSELoss()(fz, fzz)
        loss_mse_i = nn.MSELoss()(l_1, l_11)
        gravity_loss = (1-alfa)*loss_ce + alfa*(loss_mse_1+loss_mse_i)

    # print('fx shape ', fx.shape)
    # print('fz shape ', fz.shape)
    # print('y shape ', y.shape)
    # print('fzz shape ', fzz.shape)
    # print('l_1 shape ', l_1.shape)
    # print('l_11 shape ', l_11.shape)




    # total_loss = loss_ce + alfa * loss_mse
    #     loss2 = nn.CosineSimilarity(dim=1)(fz, fzz)
    #     loss2=torch.mean(loss1, dim=0)
    #     loss2 = nn.KLDivLoss()(fz, fzz)
    return gravity_loss, loss_ce, loss_mse_1, loss_mse_i
    # return loss2

def extract_centroid(model, loader , device, ls_name, ls_1_name, snapshot_name, centroid_name, centroid_1_name
                     , dbs_eps_1=0.3, dbs_smp_1=50, dbs_eps_i=0.9, dbs_smp_i=100):
    '''
    :param model: the model
    :param loader: training or testing dataloader
    :param device: device to load the model
    :param snapshot_name: name of saved parameters of model
    :param ls_name: name of the latent space file
    :param centroid_name: name of the file that contains the centroid
    :return: centroid files
    '''
    global logger
    logger.info(f'started extract_centroids {centroid_name} and {centroid_1_name}')
    ls_original = defaultdict(list)
    l_1_original = defaultdict(list)
    M_O = model
    chkp = torch.load(log_path + "{}/{}/{}/{}.pt".
                                   format(chose_dataset, chose_model, model_layer, snapshot_name))
    M_O.load_state_dict(chkp['state_dict'])

    M_O.eval()
    for (x, y) in loader:
        x = x.to(device)
        y = y.to(device)
        fx, l_1, ls = M_O(x)

        y_detached = y.detach().tolist()
        ls_detached = ls.detach().tolist()
        l_1_detached = l_1.detach().tolist()
        for key, item in zip(y_detached, ls_detached):
            ls_original[key].append(item)

        for key, item in zip(y_detached, l_1_detached):
            l_1_original[key].append(item)
        # ls_original[y.item()].append(ls.detach().squeeze().tolist())

    joblib.dump(ls_original, log_path + "{}/{}/{}/{}".
                format(chose_dataset, chose_model, model_layer, ls_name))

    joblib.dump(l_1_original, log_path + "{}/{}/{}/{}".
                format(chose_dataset, chose_model, model_layer, ls_1_name))

    # logger.info(f'start DBSCAN for {centroid_1_name}')
    # centroid_l_1_list = defaultdict(list)
    # for key, item in l_1_original.items():
    #     item_ = MinMaxScaler().fit_transform(item)
    #     dbs_model = DBSCAN(eps=dbs_eps_1, min_samples=dbs_smp_1).fit(item_)
    #     centroid_l_1_list[key] = np.mean(np.array(item)[dbs_model.labels_ == 0], axis=0)
    #     if np.all(dbs_model.labels_ == -1):
    #         print('All abnormal')
    #         print(np.unique(dbs_model.labels_, return_counts=True))
    # logger.info(f'finished DBSCAN for {centroid_1_name}')
    # joblib.dump(centroid_l_1_list, log_path + "{}/{}/{}/{}".
    #             format(chose_dataset, chose_model, model_layer, centroid_1_name))

    centroid_l_1_list = defaultdict(list)
    for key, item in l_1_original.items():
        centroid_l_1_list[key] = np.mean(np.array(item), axis=0)

    joblib.dump(centroid_l_1_list, log_path + "{}/{}/{}/{}".
                format(chose_dataset, chose_model, model_layer, centroid_1_name))

    # logger.info(f'start DBSCAN for {centroid_name}')
    # centroid_list = defaultdict(list)
    # for key, item in ls_original.items():
    #     item_ = MinMaxScaler().fit_transform(item)
    #     dbs_model = DBSCAN(eps=dbs_eps_i, min_samples=dbs_smp_i).fit(item_)
    #     centroid_list[key] = np.mean(np.array(item)[dbs_model.labels_ == 0], axis=0)
    #     if np.all(dbs_model.labels_ == -1):
    #         print('All abnormal')
    #         print(np.unique(dbs_model.labels_, return_counts=True))
    #         # print(dbs_model.labels_)
    # logger.info(f'finished DBSCAN for {centroid_name}')
    # joblib.dump(centroid_list, log_path + "{}/{}/{}/{}".
    #             format(chose_dataset, chose_model, model_layer, centroid_name))


    centroid_list = defaultdict(list)
    for key, item in ls_original.items():
        centroid_list[key] = np.mean(np.array(item), axis=0)
    joblib.dump(centroid_list, log_path + "{}/{}/{}/{}".
                format(chose_dataset, chose_model, model_layer, centroid_name))

    logger.info(f'finished extract_centroids {centroid_name} and {centroid_1_name}')
    return ls_original, centroid_list, centroid_l_1_list


#modify centroids based on my theory
####################################
# def modified_centroid(centroid_path, ls_path, track_fname, mdfy_cnt_fname, itr=1 , coef=1.5, round = 0, alfa = 0,
#                       epsilon=0.3, mu=1.0, dbs_eps = 0.3, dbs_smp=20, diam=50):
#     '''
#     :param centroid_path: indicates the name of original centroid file
#     :param ls_path: indicates the name of the latent spaces file
#     :param itr: shows the number of iteration needed for obtaining new centroid
#     :return: the new centroids
#     '''
#     global logger
#     logger.info(f'started modified_centroid {centroid_path} ')
#     # if not os.path.exists(log_path + "{}/{}/{}/{}".
#     #                             format(chose_dataset, chose_model, model_layer, track_fname)):
#     if round == 0:
#         centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
#                                     format(chose_dataset, chose_model, model_layer, centroid_path))
#         df = pd.DataFrame(centroid_list)
#         a = list(df.columns)
#         #o -> origirnal, c -> centroid, a -> alfa, e ->
#         b = [f'c{i}_{round}' for i in a]
#         # b = [f'o_c{i}_a{alfa}_e{epoch}_g{coef}' for i in a]
#         ab = dict(zip(a, b))
#         df = df.rename(columns = ab)
#         df.to_csv(log_path + "{}/{}/{}/{}".
#                   format(chose_dataset, chose_model, model_layer, track_fname),
#                   index=False)
#
#         for c_key_i, c_item_i in centroid_list.items():
#             centroid_list[c_key_i] = list(centroid_list[c_key_i])
#
#         joblib.dump(centroid_list, log_path + "{}/{}/{}/{}".
#                     format(chose_dataset, chose_model, model_layer, mdfy_cnt_fname))
#     else:
#         df = pd.read_csv(log_path + "{}/{}/{}/{}".
#                          format(chose_dataset, chose_model, model_layer, track_fname))
#
#         centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
#                                     format(chose_dataset, chose_model, model_layer, centroid_path))
#         ls_original = joblib.load(log_path + "{}/{}/{}/{}".
#                                     format(chose_dataset, chose_model, model_layer, ls_path))
#
#
#         centroid_list_temp = defaultdict(list)
#         for c_key, c_item in centroid_list.items():
#             best_item = None
#             th_distance = 0
#             item_ = MinMaxScaler().fit_transform(ls_original[c_key])
#             dbs_model = DBSCAN(eps=dbs_eps, min_samples=dbs_smp).fit(item_)
#             for index, item in enumerate(ls_original[c_key]):
#                 # print('item', np.array(item))
#                 # print('centroid', list(centroid_list.values())[0])
#                 if dbs_model.labels_[index] != 0:
#                     continue
#
#                 dis_from_others = []
#                 for a_key, b_item in centroid_list.items():
#                     if a_key == c_key:
#                         continue
#                     dis_from_others.append(np.linalg.norm(np.array(item) - np.array(b_item)))
#                 dis_from_others = np.array(dis_from_others)
#                 dis_from_others = 1e3/dis_from_others # I used 1e3 just because 1/dis_... is a very small
#                 sum_dis = np.sum(dis_from_others)
#
#                 # np_mean = np.mean([np.linalg.norm(np.array(item) - np.array(b)) for b in list(centroid_list.values())])
#                 if sum_dis > th_distance:
#                     th_distance = sum_dis
#                     best_item = item.copy()
#             # centroid_list_temp[c_key] = best_item.copy()
#             # centroid_list[c_key] = best_item.copy()
#
#             if round == 1:
#                 # direction = best_item - centroid_list[c_key]
#                 # direction_norm = direction / np.linalg.norm(direction)
#                 # step_2_dir = epsilon * direction_norm
#                 # centroid_list[c_key] += step_2_dir.copy()
#                 direction = best_item - centroid_list[c_key]
#                 # direction_norm = direction
#                 # step_2_dir = epsilon * direction
#                 direction_norm = direction / np.linalg.norm(direction)
#                 step_2_dir = diam * direction_norm
#                 centroid_list[c_key] += step_2_dir.copy()
#             else:
#                 # g_i = (df[f'c{c_key}_{round-1}'] - df[f'c{c_key}_{round-2}']).values.squeeze()/epsilon
#                 # direction = best_item - centroid_list[c_key]
#                 # direction_norm = direction / np.linalg.norm(direction)
#                 # g_i_next = mu * g_i + direction_norm
#                 # step_2_dir = epsilon * g_i_next
#                 # centroid_list[c_key] += step_2_dir.copy()
#                 g_i = (df[f'c{c_key}_{round-1}'] - df[f'c{c_key}_{round-2}']).values.squeeze()
#                 direction = best_item - centroid_list[c_key]
#                 # direction_norm = direction / np.linalg.norm(direction)
#                 g_i_next_tmp = mu * g_i + epsilon * direction
#                 g_i_next = diam *  g_i_next_tmp / np.linalg.norm(g_i_next_tmp)
#                 # g_i_next = mu * g_i + epsilon * direction
#                 # step_2_dir = epsilon * g_i_next
#                 centroid_list[c_key] += g_i_next.copy()
#
#         # centroid_list = centroid_list_temp.copy()
#         # print(np_mean)
#         # exit()
#
#         df2 = pd.DataFrame(centroid_list)
#         a = list(df2.columns)
#         #m -> modified, c -> centroid, a -> alfa, e -> epoch
#         # b = [f'm_c{i}_a{alfa}_e{epoch}_g{coef} ' for i in a]
#         b = [f'c{i}_{round}' for i in a]
#         ab = dict(zip(a, b))
#         df2 = df2.rename(columns = ab)
#         df = pd.concat([df, df2], axis=1, sort=False)
#         df.to_csv(log_path + "{}/{}/{}/{}".
#                  format(chose_dataset, chose_model, model_layer, track_fname),
#                     index=False)
#
#         joblib.dump(centroid_list, log_path + "{}/{}/{}/{}".
#                     format(chose_dataset, chose_model, model_layer, mdfy_cnt_fname))
#     logger.info(f'finished modified_centroid {centroid_path}')
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

# modify centroids based on normal vectors
# def modified_centroid(centroid_path, ls_path, track_fname, mdfy_cnt_fname, itr=4 , coef=1.5, epoch = 0, alfa = 0):
#     '''
#     :param centroid_path: indicates the name of original centroid file
#     :param ls_path: indicates the name of the latent spaces file
#     :param itr: shows the number of iteration needed for obtaining new centroid
#     :return: the new centroids
#     '''
#     #track_fname: 'centroid_tracks.csv'
#     #mdfy_cnt_fname: modified_centroids
#     if not os.path.exists(log_path + "{}/{}/{}/{}".
#                                 format(chose_dataset, chose_model, model_layer, track_fname)):
#         centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
#                                     format(chose_dataset, chose_model, model_layer, centroid_path))
#         df = pd.DataFrame(centroid_list)
#         a = list(df.columns)
#         #o -> origirnal, c -> centroid, a -> alfa, e -> epoch
#         b = [f'o_c{i}_a{alfa}_e{epoch}_g{coef}' for i in a]
#         ab = dict(zip(a, b))
#         df = df.rename(columns = ab)
#         df.to_csv(log_path + "{}/{}/{}/{}".
#                   format(chose_dataset, chose_model, model_layer, track_fname),
#                   index=False)
#
#         for c_key_i, c_item_i in centroid_list.items():
#             centroid_list[c_key_i] = list(centroid_list[c_key_i])
#
#         joblib.dump(centroid_list, log_path + "{}/{}/{}/{}".
#                     format(chose_dataset, chose_model, model_layer, mdfy_cnt_fname))
#     else:
#         df = pd.read_csv(log_path + "{}/{}/{}/{}".
#                                 format(chose_dataset, chose_model, model_layer, track_fname))
#
#         centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
#                                     format(chose_dataset, chose_model, model_layer, centroid_path))
#         ls_original = joblib.load(log_path + "{}/{}/{}/{}".
#                                     format(chose_dataset, chose_model, model_layer, ls_path))
#
#         std_list = defaultdict(list)
#         for ls_key, ls_item in ls_original.items():
#             std_list[ls_key] = np.std(list(ls_item))
#
#         d_i_j = defaultdict(dict)
#         for c_key_i, c_item_i in centroid_list.items():
#             for c_key_j, c_item_j in centroid_list.items():
#                 src = np.array(c_item_i)
#                 dst = np.array(c_item_j)
#                 d_i_j[c_key_i][c_key_j]= LA.norm(src - dst)
#
#         # print('std_list', std_list)
#         # print('d_i_j', d_i_j)
#         # exit()
#
#         bisectors = {}
#         for c_key_i, c_item_i in centroid_list.items():
#             temp_bisectors = []
#             temp_gravities = []
#             bisector = 0
#             for c_key_j, c_item_j in centroid_list.items():
#                 if c_key_i == c_key_j:
#                     continue
#                 sigma1 = std_list[c_key_i]
#                 sigma2 = std_list[c_key_j]
#                 d = (d_i_j[c_key_i][c_key_j])**2
#                 src = np.array(c_item_i)/LA.norm(c_item_i)
#                 dst = np.array(c_item_j)/LA.norm(c_item_j)
#                 temp_bisector = src - dst
#                 temp_bisectors.append(temp_bisector)
#                 temp_gravity = (sigma1*sigma2)/d
#                 temp_gravities.append(temp_gravity)
#
#             temp_gravities = coef*(temp_gravities/np.max(temp_gravities))
#             bisector = np.dot(temp_gravities, np.array(temp_bisectors))
#             # bisector += (coef*temp_bisector)/LA.norm(temp_bisector)
#             bisectors[c_key_i] = bisector.copy()
#
#         for c_key_i, c_item_i in centroid_list.items():
#             centroid_list[c_key_i] = list(np.array(centroid_list[c_key_i]) + bisectors[c_key_i])
#
#         df2 = pd.DataFrame(centroid_list)
#         a = list(df2.columns)
#         #m -> modified, c -> centroid, a -> alfa, e -> epoch
#         b = [f'm_c{i}_a{alfa}_e{epoch}_g{coef}' for i in a]
#         ab = dict(zip(a, b))
#         df2 = df2.rename(columns = ab)
#         df = pd.concat([df, df2], axis=1, sort=False)
#         df.to_csv(log_path + "{}/{}/{}/{}".
#                  format(chose_dataset, chose_model, model_layer, track_fname),
#                     index=False)
#
#         joblib.dump(centroid_list, log_path + "{}/{}/{}/{}".
#                     format(chose_dataset, chose_model, model_layer, mdfy_cnt_fname))
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

# modify centroids based on normal vectors
def modified_centroid(centroid_path, ls_path, track_fname, mdfy_cnt_fname, itr=1 , coef=1.5, round = 0, alfa = 0,
                      epsilon=0.3, mu=1.0, dbs_eps = 0.3, dbs_smp=20, diam=50):
    '''
    :param centroid_path: indicates the name of original centroid file
    :param ls_path: indicates the name of the latent spaces file
    :param itr: shows the number of iteration needed for obtaining new centroid
    :return: the new centroids
    '''
    global logger
    logger.info(f'started modified_centroid {centroid_path} ')

    if round == 0:
        centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
                                    format(chose_dataset, chose_model, model_layer, centroid_path))
        df = pd.DataFrame(centroid_list)
        a = list(df.columns)
        #o -> origirnal, c -> centroid, a -> alfa, e ->
        b = [f'c{i}_{round}' for i in a]
        # b = [f'o_c{i}_a{alfa}_e{epoch}_g{coef}' for i in a]
        ab = dict(zip(a, b))
        df = df.rename(columns = ab)
        df.to_csv(log_path + "{}/{}/{}/{}".
                  format(chose_dataset, chose_model, model_layer, track_fname),
                  index=False)

        for c_key_i, c_item_i in centroid_list.items():
            centroid_list[c_key_i] = list(centroid_list[c_key_i])

        joblib.dump(centroid_list, log_path + "{}/{}/{}/{}".
                    format(chose_dataset, chose_model, model_layer, mdfy_cnt_fname))
    else:
        df = pd.read_csv(log_path + "{}/{}/{}/{}".
                         format(chose_dataset, chose_model, model_layer, track_fname))

        centroid_list = joblib.load(log_path + "{}/{}/{}/{}".
                                    format(chose_dataset, chose_model, model_layer, centroid_path))
        ls_original = joblib.load(log_path + "{}/{}/{}/{}".
                                    format(chose_dataset, chose_model, model_layer, ls_path))

        std_list = defaultdict(list)
        for ls_key, ls_item in ls_original.items():
            std_list[ls_key] = np.std(list(ls_item))

        d_i_j = defaultdict(dict)
        for c_key_i, c_item_i in centroid_list.items():
            for c_key_j, c_item_j in centroid_list.items():
                src = np.array(c_item_i)
                dst = np.array(c_item_j)
                d_i_j[c_key_i][c_key_j]= LA.norm(src - dst)

        # print('std_list', std_list)
        # print('d_i_j', d_i_j)
        # exit()

        bisectors = {}
        for c_key_i, c_item_i in centroid_list.items():
            temp_bisectors = []
            temp_gravities = []
            bisector = 0
            for c_key_j, c_item_j in centroid_list.items():
                if c_key_i == c_key_j:
                    continue
                sigma1 = std_list[c_key_i]
                sigma2 = std_list[c_key_j]
                d = (d_i_j[c_key_i][c_key_j])**2
                # src = np.array(c_item_i)/LA.norm(c_item_i)
                # dst = np.array(c_item_j)/LA.norm(c_item_j)
                src = np.array(c_item_i)
                dst = np.array(c_item_j)
                temp_bisector = src - dst
                temp_bisectors.append(temp_bisector)
                temp_gravity = (sigma1*sigma2)/d
                temp_gravities.append(temp_gravity)


            normalize_bisecotrs = [x/np.linalg.norm(x) for x in temp_bisectors]
            bisector = np.dot(temp_gravities, np.array(normalize_bisecotrs))
            # bisector = np.dot(temp_gravities, np.array(temp_bisectors))

            bisectors[c_key_i] = bisector.copy()

        ## this section for normalizing each vector based on the force magnitiude
        b = np.array(list(bisectors.values()))
        c = np.linalg.norm(b, axis=1)
        d = np.max(c)

        for c_key_i, c_item_i in centroid_list.items():
            if round == 1:
                centroid_list[c_key_i] += (diam * epsilon *  bisectors[c_key_i] / d).copy()
            else:
                g_i = (df[f'c{c_key_i}_{round-1}'] - df[f'c{c_key_i}_{round-2}']).values.squeeze()
                g_i_next = mu * g_i + epsilon * diam * bisectors[c_key_i] / d
                # g_i_next = diam *  g_i_next_tmp / np.linalg.norm(g_i_next_tmp)
                centroid_list[c_key_i] += g_i_next.copy()


        # for c_key_i, c_item_i in centroid_list.items():
        #     if round == 1:
        #         direction = bisectors[c_key_i]
        #         direction_norm = direction / np.linalg.norm(direction)
        #         step_2_dir = diam * epsilon *  direction_norm
        #         centroid_list[c_key_i] += step_2_dir.copy()
        #     else:
        #         g_i = (df[f'c{c_key_i}_{round-1}'] - df[f'c{c_key_i}_{round-2}']).values.squeeze()
        #         g_i_next_tmp = mu * g_i + epsilon * bisectors[c_key_i]
        #         g_i_next = diam *  g_i_next_tmp / np.linalg.norm(g_i_next_tmp)
        #         centroid_list[c_key_i] += g_i_next.copy()

        # for c_key_i, c_item_i in centroid_list.items():
        #     centroid_list[c_key_i] = list(np.array(centroid_list[c_key_i]) + bisectors[c_key_i])

        df2 = pd.DataFrame(centroid_list)
        a = list(df2.columns)
        b = [f'c{i}_{round}' for i in a]
        ab = dict(zip(a, b))
        df2 = df2.rename(columns = ab)
        df = pd.concat([df, df2], axis=1, sort=False)
        df.to_csv(log_path + "{}/{}/{}/{}".
                 format(chose_dataset, chose_model, model_layer, track_fname),
                    index=False)

        joblib.dump(centroid_list, log_path + "{}/{}/{}/{}".
                    format(chose_dataset, chose_model, model_layer, mdfy_cnt_fname))
    logger.info(f'finished modified_centroid {centroid_path}')
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

def vis(ls_name, component=2, technique='tsne', path=None, maped=None):
    if path == None:
        ls_original = ls_path
    else:
        ls_original = joblib.load(log_path + "{}/{}/{}/{}".
                                  format(chose_dataset, chose_model, model_layer, ls_name))

    # ls_original = joblib.load(ls_path)
    lbl = []
    color = []
    classes = [x for x in range(10)]
    colors = ['pink', 'blue', 'yellow',
              'red', 'green', 'orange',
              'purple', 'brown', 'cyan', 'olive']
    for index, (key, item) in enumerate(ls_original.items()):
        # new_lbl = [key + 100] * len(item)
        #         new_color = [key + 10]*len(item)
        new_lbl = [key] * len(item)
        lbl += new_lbl.copy()
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

    if technique == 'pca':
        pca = decomposition.PCA(n_components=2, random_state=3)
        pca.fit(out_arr)
        tsne_data = pca.transform(out_arr)

        tsne_data = np.vstack((tsne_data.T, np.array(lbl))).T
        tsne_df = pd.DataFrame(data=tsne_data, columns=['dim_1', 'dim_2', 'lbl'])

        fig, ax = plt.subplots()
        for lbl_ , color_ in zip(classes, colors):
            ax.scatter(tsne_df[tsne_df['lbl']==lbl_]['dim_1'].values,
                       tsne_df[tsne_df['lbl']==lbl_]['dim_2'].values,
                       c=color_,
                       label=lbl_,
                       alpha=0.5)


        ax.legend(bbox_to_anchor=(1., 1.03), loc="upper left")

        fig.savefig(log_path + "{}/{}/{}/{}.png".
                    format(chose_dataset, chose_model, model_layer, ls_name), bbox_inches='tight', dpi=300)
        # plt.show()

# def vis(ls_name, component=2, technique='tsne', path=None, maped=None):
#     if path == None:
#         ls_original = ls_path
#     else:
#         ls_original = joblib.load(log_path + "{}/{}/{}/{}".
#                                   format(chose_dataset, chose_model, model_layer, ls_name))
#
#     # ls_original = joblib.load(ls_path)
#     lbl = []
#     color = []
#     colors_ = np.linspace(0xf00000, 0xF0f000, 10, dtype=int)
#     colors = np.array(['#' + str(hex(clr))[2:] for clr in colors_])
#     for index, (key, item) in enumerate(ls_original.items()):
#         new_lbl = [key + 100] * len(item)
#         #         new_color = [key + 10]*len(item)
#         lbl += new_lbl.copy()
#         if index == 0:
#             out_arr = np.array(item)
#         else:
#             new_item = np.array(item)
#             out_arr = np.vstack((out_arr, new_item))
#
#     if technique == 'tsne':
#         tsne_model = TSNE(n_components=component, random_state=SEED)
#         tsne_data_ = tsne_model.fit_transform(out_arr)
#         tsne_data = tsne_data_.copy()
#
#         tsne_data = np.vstack((tsne_data.T, np.array(lbl), np.array(color))).T
#         tsne_df = pd.DataFrame(data=tsne_data, columns=['dim_1', 'dim_2', 'lbl', 'color'])
#         sns.FacetGrid(tsne_df, hue='color', size=6).map(plt.scatter, 'dim_1', 'dim_2', 'color').add_legend()
#         plt.savefig(log_path + "{}/{}/{}/{}.png".
#                                 format(chose_dataset, chose_model, model_layer, ls_name))
#
#     if technique == 'pca':
#         pca = decomposition.PCA(n_components=2, random_state=3)
#         pca.fit(out_arr)
#         tsne_data = pca.transform(out_arr)
#
#         tsne_data = np.vstack((tsne_data.T, np.array(lbl))).T
#         tsne_df = pd.DataFrame(data=tsne_data, columns=['dim_1', 'dim_2', 'lbl'])
#         if np.all(maped == None):
#             sns.FacetGrid(tsne_df, hue='lbl', size=6).map(plt.scatter, 'dim_1', 'dim_2', 'lbl').add_legend()
#         else:
#             tsne_df.plot.scatter(x='dim_1', y='dim_2', c=maped)
#         plt.savefig(log_path + "{}/{}/{}/{}.png".
#                     format(chose_dataset, chose_model, model_layer, ls_name))
#         # plt.show()

# # Mean and Standard Deiation of the Dataset
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]


def normalize(t, mean, std, dataset):

    if dataset == 'mnist':
        t[:, 0, :] = (t[:, 0, :] - mean[0]) / std[0]
    if dataset == 'cifar10':
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def un_normalize(t, mean, std, dataset):
    if dataset == 'mnist':
        t[:, 0, :] = (t[:, 0, :] * std[0]) + mean[0]
    if dataset == 'cifar10':
        t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
        t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
        t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t



# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters, mean, std, dataset='mnist', black_box=False):
    # adv = img.detach()
    adv = img.clone()
    adv.requires_grad = True

    if attack_type == 'fgsm' or attack_type == 'cw':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

        noise = 0

    if dataset == 'mnist' or dataset == 'fmnist':
        for j in range(iterations):
            if black_box == False:
                out_adv, _, _ = model(normalize(adv.clone(), mean, std, dataset))
            else:
                out_adv = model(normalize(adv.clone(), mean, std, dataset))
            loss = criterion(out_adv, label)
            # loss = F.nll_loss(out_adv, label)
            loss.backward()

            if attack_type == 'mim':
                adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
                adv.grad = adv.grad / adv_mean
                noise = noise + adv.grad
            else:
                noise = adv.grad

            # Optimization step
            adv.data = adv.data + step * noise.sign()
            #        adv.data = adv.data + step * adv.grad.sign()

            if attack_type == 'pgd':
                adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
                adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)

            adv.data.clamp_(0.0, 1.0)

            if attack_type == 'cw':
                inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
                              max((1 - m) / s for m, s in zip(mean, std)))
                # an untargeted adversary
                adversary = cw.L2Adversary(targeted=False,
                                           confidence=0.0,
                                           search_steps=iters,
                                           box=inputs_box,
                                           optimizer_lr=5e-4)

                adv = adversary(model, img, label, to_numpy=False)

            adv.grad.data.zero_()
    else:
        for j in range(iterations):
            if black_box == False:
                out_adv, _, _ = model(normalize(adv.clone(), mean, std, dataset))
            else:
                out_adv = model(normalize(adv.clone(), mean, std, dataset))
            loss = criterion(out_adv, label)
            loss.backward()

            if attack_type == 'mim':
                adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
                adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
                adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
                adv.grad = adv.grad / adv_mean
                noise = noise + adv.grad
            else:
                noise = adv.grad

            # Optimization step
            adv.data = adv.data + step * noise.sign()
            #        adv.data = adv.data + step * adv.grad.sign()

            if attack_type == 'pgd':
                adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
                adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
            adv.data.clamp_(0.0, 1.0)

            if attack_type == 'cw':
                inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
                              max((1 - m) / s for m, s in zip(mean, std)))
                # an untargeted adversary
                adversary = cw.L2Adversary(targeted=False,
                                           confidence=0.0,
                                           search_steps=iters,
                                           box=inputs_box,
                                           optimizer_lr=1e-2)

                adv = adversary(model, img, label, to_numpy=False)

            adv.grad.data.zero_()

    return adv.detach()


