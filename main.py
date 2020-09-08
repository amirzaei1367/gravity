 ## imported module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wget
from sklearn.externals import joblib
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
from math import log10
import pytorch_ssim
# import cv2

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
# import joblib

# import joblib

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn import svm
import pandas as pd

import random
import time
import logging
import os
import json
import copy

import numpy as np
import matplotlib.pylab as plt

from model import *
from settings import *
from helpers import *

from advertorch.attacks import FGSM
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.attacks import LinfBasicIterativeAttack
from advertorch.attacks import LinfMomentumIterativeAttack

from advertorch.utils import predict_from_logits

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


### setting up the log
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
## setting up the dataloaders
def cifar10_loader(logger, batch_size=1):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_transformer = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    ## transformer for the attack
    adv_transformer = transforms.Compose([
        transforms.ToTensor()])

    train_data = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transformer)
    test_data = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=dev_transformer)
    adv_data = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=adv_transformer)

    train_iterator = torch.utils.data.DataLoader(train_data,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=8,
                                                 pin_memory=True)

    test_iterator = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True,
                                                drop_last=True)

    adv_iterator = torch.utils.data.DataLoader(adv_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True,
                                                drop_last=True)

    return train_iterator, test_iterator, adv_iterator, len(train_data), len(test_data)

def mnist_loaders(logger, batch_size=1):
    data_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307],
                             std=[0.3081])
    ])

    adv_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])

    train_data = datasets.MNIST(dataset_path,
                                train=True,
                                download=False,
                                transform=data_transforms)

    train_iterator = torch.utils.data.DataLoader(train_data,
                                                 shuffle=True,
                                                 batch_size=batch_size)

    test_data = datasets.MNIST(dataset_path,
                               train=False,
                               download=False,
                               transform=data_transforms)

    test_iterator = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

    adv_data = datasets.MNIST(dataset_path,
                               train=False,
                               download=False,
                               transform=adv_transforms)

    adv_iterator = torch.utils.data.DataLoader(adv_data, shuffle=False, batch_size=batch_size)

    return train_iterator, test_iterator, adv_iterator, len(train_data), len(test_data)

def svhn_loaders(logger, batch_size=1):
    data_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4378, 0.4439, 0.4729],
                             std=[0.1980, 0.2011, 0.1971])
    ])

    adv_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])

    train_data = datasets.SVHN(dataset_path,
                                split='train',
                                download=False,
                                transform=data_transforms)

    train_iterator = torch.utils.data.DataLoader(train_data,
                                                 shuffle=True,
                                                 batch_size=batch_size)

    test_data = datasets.SVHN(dataset_path,
                               split='test',
                               download=False,
                               transform=data_transforms)

    test_iterator = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

    adv_data = datasets.SVHN(dataset_path,
                               split='test',
                               download=False,
                               transform=adv_transforms)

    adv_iterator = torch.utils.data.DataLoader(adv_data, shuffle=False, batch_size=batch_size)

    return train_iterator, test_iterator, adv_iterator, len(train_data), len(test_data)

def fmnist_loaders(logger, batch_size=1):
    data_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307],
                             std=[0.3081])
    ])

    adv_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()])

    train_data = datasets.FashionMNIST(dataset_path,
                                train=True,
                                download=False,
                                transform=data_transforms)

    train_iterator = torch.utils.data.DataLoader(train_data,
                                                 shuffle=True,
                                                 batch_size=batch_size)

    test_data = datasets.FashionMNIST(dataset_path,
                               train=False,
                               download=False,
                               transform=data_transforms)

    test_iterator = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

    adv_data = datasets.FashionMNIST(dataset_path,
                               train=False,
                               download=False,
                               transform=adv_transforms)

    adv_iterator = torch.utils.data.DataLoader(adv_data, shuffle=False, batch_size=batch_size)

    return train_iterator, test_iterator, adv_iterator, len(train_data), len(test_data)

# def fgsm_adv_training_kd_loop(model, logger, device, optimizer, criterion,
#                                 train_iterator, test_iterator, index, alfa=0.1,
#                                 coef=2, epochs=CENT_SWITCH, epsilon=0.3, mu=1.0):
#     logger.info(f'The model has {count_parameters(model):,} trainable parameters')
#
#     metrics = {}
#     history_gravity_loss = []
#     history_loss_ce = []
#     history_loss_mse_1 = []
#     history_loss_mse_i = []
#     history_acc = []
#     best_test_acc = 0.0
#     for epoch in range(epochs):
#         if (epoch == (epochs-1)):
#             extract_centroid(model=model,
#                              loader=train_iterator,
#                              device=device,
#                              snapshot_name=f'gravity_best_{alfa}_{index}',
#                              ls_name=f'ls_gravity_{alfa}_{index}',
#                              centroid_name=f'gravity_centroids_{alfa}_{index}',
#                              ls_1_name=f'ls_1_gravity_{alfa}_{index}',
#                              centroid_1_name=f'gravity_1_centroids_{alfa}_{index}',
#                              dbs_eps=50,
#                              dbs_smp=10)
#
#             modified_centroid(centroid_path=f'gravity_centroids_{alfa}_{index}',
#                               ls_path=f'ls_gravity_{alfa}_{index}',
#                               itr=1,
#                               coef=coef,
#                               track_fname='centroid_tracks.csv',
#                               mdfy_cnt_fname='modified_centroids',
#                               round=index,
#                               alfa=alfa,
#                               epsilon=epsilon,
#                               mu=mu,
#                               dbs_eps=50,
#                               dbs_smp=10,
#                               diam=)
#
#             modified_centroid(centroid_path=f'gravity_1_centroids_{alfa}_{index}',
#                               ls_path=f'ls_1_gravity_{alfa}_{index}',
#                               itr=1,
#                               coef=coef,
#                               track_fname='centroid_1_tracks.csv',
#                               mdfy_cnt_fname='modified_1_centroids',
#                               round=index,
#                               alfa=alfa,
#                               epsilon=epsilon,
#                               mu=mu,
#                               dbs_eps=50,
#                               dbs_smp=10,
#                               diam=)
#
#         start_time = time.time()
#
#         train_loss, train_acc, loss_ce, loss_mse_1, loss_mse_i = train_kd(model=model,
#                                                                  iterator=train_iterator,
#                                                                  optimizer=optimizer,
#                                                                  criterion=criterion,
#                                                                  device=device,
#                                                                  centroid='modified_centroids',
#                                                                  centroid_1='modified_1_centroids',
#                                                                  alfa=alfa,
#                                                                  adv_trainer='fgsm')
#
#         test_loss, test_acc = evaluate(model=model,
#                                        iterator=test_iterator,
#                                        criterion=criterion,
#                                        device=device)
#
#         history_gravity_loss.append(train_loss)
#         history_loss_ce.append(loss_ce)
#         history_loss_mse_1.append(loss_mse_1)
#         history_loss_mse_i.append(loss_mse_i)
#         history_acc.append(train_acc)
#         if test_acc >= best_test_acc:
#             best_test_acc = test_acc
#             logger.info("new best at epoch {} with acc {}".format(epoch, test_acc))
#             logger.info("new best at epoch {} with acc {}".format(epoch, test_acc))
#             torch.save({'state_dict': model.state_dict()},
#                        log_path + "{}/{}/{}/gravity_best_{}_{}.pt".format(chose_dataset,
#                                                                        chose_model, model_layer, alfa, index))
#
#             var = {'epoch': epoch,
#                    'test_acc': test_acc,
#                    'test_loss': test_loss,
#                    'train_acc': train_acc,
#                    'train_loss': train_loss}
#             with open(log_path + "{}/{}/{}/gravity_score.json".format(chose_dataset,
#                                                                       chose_model,
#                                                                       model_layer),"w") as p:
#                 json.dump(var, p)
#
#         end_time = time.time()
#
#         epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#
#         logger.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
#         logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
#         logger.info(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')
#
#
#     if not os.path.exists(log_path + "{}/{}/{}/metrics.json".
#             format(chose_dataset, chose_model, model_layer)):
#
#         metrics['gravity_loss'] = history_gravity_loss
#         metrics['loss_ce'] = history_loss_ce
#         metrics['loss_mse_1'] = history_loss_mse_1
#         metrics['loss_mse_i'] = history_loss_mse_i
#         metrics['train_acc'] = history_acc
#         with open(log_path + "{}/{}/{}/metrics.json".
#                 format(chose_dataset, chose_model, model_layer), "w") as p:
#             json.dump(metrics, p, indent=4)
#     else:
#         with open(log_path + "{}/{}/{}/metrics.json".
#                 format(chose_dataset, chose_model, model_layer), 'r+') as p:
#             metrics=json.load(p)
#             metrics['gravity_loss']+=(history_gravity_loss)
#             metrics['loss_ce'] += history_loss_ce
#             metrics['loss_mse_1'] += history_loss_mse_1
#             metrics['loss_mse_i'] += history_loss_mse_i
#             metrics['train_acc'] += history_acc
#             # p.seek(0)
#             json.dump(metrics, open(log_path + "{}/{}/{}/metrics.json".
#                 format(chose_dataset, chose_model, model_layer), 'w'), indent=4)
#             # p.truncate()
#
#         # os.remove(log_path + "{}/{}/{}/metrics.json".
#         #         format(chose_dataset, chose_model, model_layer))
#         # with open(log_path + "{}/{}/{}/metrics.json".
#         #         format(chose_dataset, chose_model, model_layer), "w") as p:
#         #     json.dump(metrics, p, indent=4)
#
#
#     return var
#
#
# def pgd_adv_training_kd_loop(model, logger, device, optimizer, criterion,
#                             train_iterator, test_iterator, index, alfa=0.1,
#                             coef=2, epochs=CENT_SWITCH, epsilon=0.3, mu=1.0):
#     pass
## training loops

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    global logger
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif (np.abs(score - self.best_score) < self.delta) or ((score - self.best_score) > self.delta):
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score < self.best_score:
            self.best_score = score
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     self.val_loss_min = val_loss

def train(model, iterator, optimizer, criterion, device, black_box=False):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)


        optimizer.zero_grad()

        if black_box == True:
            fx = model(x)
        else:
            fx, _, _ = model(x)

        loss = criterion(fx, y)
        acc = calculate_accuracy(fx, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_loop(model, logger, device, optimizer, scheduler, criterion, train_iterator,
               test_iterator, chk_fname = 'original_best', score_fname = "original_score", black_box=False):
    # model = LeNet(OUTPUT_DIM, INPUT_DIM)
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    # optimizer = optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss()
    # # model = model.to(device)
    # criterion = criterion.to(device)

    # scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    best_test_acc = 0.0

    for epoch in range(EPOCHS):
        # print('idx epoch {}/{} best_loss {}'.format(epoch, EPOCHS, best_test_loss))
        scheduler.step()
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device, black_box=black_box)
        test_loss, test_acc, pred, gt = evaluate(model, test_iterator, criterion, device, black_box=black_box)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            logger.info("new best at epoch {} with acc {}".format(epoch, test_acc))
            logger.info("new best at epoch {} with acc {}".format(epoch, test_loss))
            torch.save({'state_dict': model.state_dict()}, log_path + "{}/{}/{}/{}.pt".
                       format(chose_dataset, chose_model, model_layer, chk_fname))

            var = {'epoch': epoch,
                   'test_acc': test_acc,
                   'test_loss': test_loss,
                   'train_acc': train_acc,
                   'train_loss': train_loss}


            with open(log_path + "{}/{}/{}/{}.json".format(chose_dataset, chose_model, model_layer, score_fname ),
                      "w") as p:
                json.dump(var, p)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        logger.info(f'Epoch: {epoch + 1:02}, LR: {scheduler.get_lr()} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        logger.info(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')

    return var

##trainig_kd loop
def train_kd(model, iterator, optimizer, criterion, device, centroid, centroid_1, alfa, adv_trainer='fgsm'):

    if chose_dataset == 'cifar10':
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    if chose_dataset == 'cifar100':
        preprocessing = dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    if chose_dataset == 'svhn':
        preprocessing = dict(mean=[0.4378, 0.4439, 0.4729], std=[0.1980, 0.2011, 0.1971])

    if chose_dataset == 'mnist':
        preprocessing = dict(mean=[0.1307], std=[0.3081])

    if chose_dataset == 'fmnist':
        preprocessing = dict(mean=[0.1307], std=[0.3081])


    epoch_gravity_loss = 0
    epoch_loss_mse_i = 0
    epoch_loss_ce = 0
    epoch_loss_mse_1 = 0
    epoch_acc = 0




    centroid_list = joblib.load(log_path + "{}/{}/{}/{}".format(chose_dataset,
                                                                chose_model,
                                                                model_layer,
                                                                centroid))
    centroid_1_list = joblib.load(log_path + "{}/{}/{}/{}".format(chose_dataset,
                                                                  chose_model,
                                                                  model_layer,
                                                                  centroid_1))
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        model.eval()

        if adv_trainer == None:
            pass
        if adv_trainer == 'fgsm':
            if chose_model == 'lenet':
                model2 = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
            if chose_model == 'resnet':
                model2 = resnet_fb(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
            model2.to(device)
            model2.load_state_dict(model.state_dict(), strict=False)

            eps = np.random.uniform(0.02, 0.05)
            # chkp2 = torch.load(log_path + "{}/{}/{}/gravity_best_{}.pt".
            #                    format(chose_dataset, chose_model, model_layer, alfa), map_location=device)
            # model.load_state_dict(chkp2['state_dict'])
            attack_M_G = FGSM(predict=model2,
                              loss_fn=criterion,
                              eps=eps,
                              clip_min=0.0,
                              clip_max=1.0,
                              targeted=False)
            adv = attack_M_G.perturb(x, y)
            adv = normalize(adv, mean= preprocessing['mean'], std=preprocessing['std'], dataset=chose_dataset)
            adv = adv.to(device)
            x = torch.cat((x, adv), 0)
            y = torch.cat((y, y))

        if adv_trainer == 'pgd':
            if chose_model == 'lenet':
                model2 = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
            if chose_model == 'resnet':
                model2 = resnet_fb(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
            model2.to(device)
            model2.load_state_dict(model.state_dict(), strict=False)
            eps = np.random.uniform(0.02, 0.05)
            # chkp2 = torch.load(log_path + "{}/{}/{}/gravity_best_{}.pt".
            #                    format(chose_dataset, chose_model, model_layer, alfa), map_location=device)
            # model.load_state_dict(chkp2['state_dict'])
            attack_M_G = LinfPGDAttack(predict=model2,
                                       loss_fn=criterion,
                                       eps=eps,
                                       nb_iter=10,
                                       eps_iter=eps / 10,
                                       clip_min=0.0,
                                       clip_max=1.0,
                                       targeted=False)
            adv = attack_M_G.perturb(x, y)
            adv = normalize(adv, mean= preprocessing['mean'], std=preprocessing['std'], dataset=chose_dataset)
            adv = adv.to(device)
            x = torch.cat((x, adv), 0)
            y = torch.cat((y, y))

        model.train()
        optimizer.zero_grad()
        fx, l_1, fz = model(x)

        fzz = torch.tensor([centroid_list[lbl.item()] for lbl in y]).to(device, dtype= torch.float)
        l_11 = torch.tensor([centroid_1_list[lbl.item()] for lbl in y]).to(device, dtype= torch.float)
        # gravity_loss, loss_ce, loss_mse_1, loss_mse_i
        gravity_loss, loss_ce, loss_mse_1, loss_mse_i = criterion_kd(fx=fx, #softmax probability
                                                                     fz=fz, #latent space logits at epoch i at layer j
                                                                     y=y, #ground truth
                                                                     fzz=fzz, #latent space logits at epoch i-1 at layer j
                                                                     l_1=l_1, #latent space logits at epoch i at layer -1
                                                                     l_11=l_11, #latent space logits at epoch i-1 at layer -1
                                                                     alfa=alfa)

        acc = calculate_accuracy(fx, y)

        gravity_loss.backward()
        optimizer.step()

        epoch_loss_ce += loss_ce.item()
        epoch_loss_mse_1 += loss_mse_1.item()
        epoch_loss_mse_i += loss_mse_i.item()
        epoch_gravity_loss += gravity_loss.item()
        epoch_acc += acc.item()

    mini_batch = len(iterator)
    l1 = epoch_gravity_loss / mini_batch
    l2 = epoch_acc / mini_batch
    l3 = epoch_loss_ce / mini_batch
    l4 = epoch_loss_mse_1 / mini_batch
    l5 = epoch_loss_mse_i / mini_batch

    return l1, l2, l3, l4, l5

def train_kd_loop(model, logger, device, optimizer, scheduler, criterion,
                  train_iterator, test_iterator, index, alfa=0.1,
                  coef=2, epochs=CENT_SWITCH, epsilon_1=0.3, mu_1=1.0, epsilon_i=0.9, mu_i=0.0,
                  diam_1=50, dbs_eps_1=0.3, dbs_smp_1=50, dbs_eps_i=0.9, dbs_smp_i=30, diam_i=100,  adv_trainer=None):
    # model = LeNet(OUTPUT_DIM)
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    # scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    metrics = {}
    history_gravity_loss = []
    history_loss_ce = []
    history_loss_mse_1 = []
    history_loss_mse_i = []
    history_acc = []
    best_test_acc = 0.0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10,
                                   verbose=True,
                                   delta=0.1,
                                   path=None)

    for epoch in range(epochs):
        scheduler.step()
        start_time = time.time()

        if adv_trainer == 'fgsm':
            train_loss, train_acc, loss_ce, loss_mse_1, loss_mse_i = train_kd(model=model,
                                                                     iterator=train_iterator,
                                                                     optimizer=optimizer,
                                                                     criterion=criterion,
                                                                     device=device,
                                                                     centroid='modified_centroids',
                                                                     centroid_1='modified_1_centroids',
                                                                     alfa=alfa,
                                                                     adv_trainer='fgsm')
        if adv_trainer == 'pgd':
            train_loss, train_acc, loss_ce, loss_mse_1, loss_mse_i = train_kd(model=model,
                                                                     iterator=train_iterator,
                                                                     optimizer=optimizer,
                                                                     criterion=criterion,
                                                                     device=device,
                                                                     centroid='modified_centroids',
                                                                     centroid_1='modified_1_centroids',
                                                                     alfa=alfa,
                                                                     adv_trainer='pgd')
        if adv_trainer == None:
            train_loss, train_acc, loss_ce, loss_mse_1, loss_mse_i = train_kd(model=model,
                                                                     iterator=train_iterator,
                                                                     optimizer=optimizer,
                                                                     criterion=criterion,
                                                                     device=device,
                                                                     centroid='modified_centroids',
                                                                     centroid_1='modified_1_centroids',
                                                                     alfa=alfa,
                                                                     adv_trainer=None)



        test_loss, test_acc, pred, gt = evaluate(model=model,
                                       iterator=test_iterator,
                                       criterion=criterion,
                                       device=device)

        history_gravity_loss.append(train_loss)
        history_loss_ce.append(loss_ce)
        history_loss_mse_1.append(loss_mse_1)
        history_loss_mse_i.append(loss_mse_i)
        history_acc.append(train_acc)


        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            logger.info("new best at epoch {} with acc {}".format(epoch, test_acc))
            logger.info("new best at epoch {} with acc {}".format(epoch, test_acc))
            torch.save({'state_dict': model.state_dict()},
                       log_path + "{}/{}/{}/gravity_best_{}_{}.pt".format(chose_dataset,
                                                                       chose_model, model_layer, alfa, index))

            var = {'epoch': epoch,
                   'test_acc': test_acc,
                   'test_loss': test_loss,
                   'train_acc': train_acc,
                   'train_loss': train_loss}

            conf_mtx = confusion_matrix(gt, pred)
            joblib.dump(conf_mtx, log_path + "{}/{}/{}/{}_{}.mtx".
                        format(chose_dataset, chose_model, model_layer, alfa, index))

            # with open(log_path + "{}/{}/{}/{}_{}.mtx".format(chose_dataset, chose_model, model_layer, alfa, index),
            #           "w") as p:
            #     json.dump(conf_mtx, p)


            with open(log_path + "{}/{}/{}/gravity_score.json".format(chose_dataset,
                                                                      chose_model,
                                                                      model_layer),"w") as p:
                json.dump(var, p)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        logger.info(f'Epoch: {epoch + 1:02}, LR: {scheduler.get_lr()} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        logger.info(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')

        early_stopping(train_loss, model)

        if early_stopping.early_stop:
            extract_centroid(model=model,
                             loader=train_iterator,
                             device=device,
                             snapshot_name=f'gravity_best_{alfa}_{index}',
                             ls_name=f'ls_gravity_{alfa}_{index}',
                             centroid_name=f'gravity_centroids_{alfa}_{index}',
                             ls_1_name=f'ls_1_gravity_{alfa}_{index}',
                             centroid_1_name=f'gravity_1_centroids_{alfa}_{index}',
                             dbs_eps_1=dbs_eps_1,
                             dbs_smp_1=dbs_smp_1,
                             dbs_eps_i=dbs_eps_i,
                             dbs_smp_i=dbs_smp_i)

            modified_centroid(centroid_path=f'gravity_centroids_{alfa}_{index}',
                              ls_path=f'ls_gravity_{alfa}_{index}',
                              itr=1,
                              coef=coef,
                              track_fname='centroid_tracks.csv',
                              mdfy_cnt_fname='modified_centroids',
                              round=index,
                              alfa=alfa,
                              epsilon=epsilon_i,
                              mu=mu_i,
                              dbs_eps=dbs_eps_i,
                              dbs_smp=dbs_smp_i,
                              diam=diam_i)

            modified_centroid(centroid_path=f'gravity_1_centroids_{alfa}_{index}',
                              ls_path=f'ls_1_gravity_{alfa}_{index}',
                              itr=1,
                              coef=coef,
                              track_fname='centroid_1_tracks.csv',
                              mdfy_cnt_fname='modified_1_centroids',
                              round=index,
                              alfa=alfa,
                              epsilon=epsilon_1,
                              mu=mu_1,
                              dbs_eps=dbs_eps_1,
                              dbs_smp=dbs_smp_1,
                              diam=diam_1)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    if not os.path.exists(log_path + "{}/{}/{}/metrics.json".
            format(chose_dataset, chose_model, model_layer)):

        metrics['gravity_loss'] = history_gravity_loss
        metrics['loss_ce'] = history_loss_ce
        metrics['loss_mse_1'] = history_loss_mse_1
        metrics['loss_mse_i'] = history_loss_mse_i
        metrics['train_acc'] = history_acc
        with open(log_path + "{}/{}/{}/metrics.json".
                format(chose_dataset, chose_model, model_layer), "w") as p:
            json.dump(metrics, p, indent=4)
    else:
        with open(log_path + "{}/{}/{}/metrics.json".
                format(chose_dataset, chose_model, model_layer), 'r+') as p:
            metrics=json.load(p)
            metrics['gravity_loss']+=(history_gravity_loss)
            metrics['loss_ce'] += history_loss_ce
            metrics['loss_mse_1'] += history_loss_mse_1
            metrics['loss_mse_i'] += history_loss_mse_i
            metrics['train_acc'] += history_acc
            # p.seek(0)
            json.dump(metrics, open(log_path + "{}/{}/{}/metrics.json".
                format(chose_dataset, chose_model, model_layer), 'w'), indent=4)
            # p.truncate()

        # os.remove(log_path + "{}/{}/{}/metrics.json".
        #         format(chose_dataset, chose_model, model_layer))
        # with open(log_path + "{}/{}/{}/metrics.json".
        #         format(chose_dataset, chose_model, model_layer), "w") as p:
        #     json.dump(metrics, p, indent=4)


    return var



def main_trainer():

    logger.info("****************************")
    logger.info("SETTING INFO:")
    logger.info(f"Dataset: {chose_dataset}")
    logger.info(f"Model: {chose_model}")
    logger.info(f"surgery layer: {model_layer}")
    logger.info(f"EPOCHS: {EPOCHS}")
    logger.info(f"CENT_SWITCH: {CENT_SWITCH}")
    logger.info(f"epsilon_1: {EPSILON_1}")
    logger.info(f"mu_1: {MU_1}")
    logger.info(f"epsilon_i: {EPSILON_i}")
    logger.info(f"mu_i: {MU_i}")
    logger.info(f"diam_i: {Circle_Diam_i}")
    logger.info(f"diam_1: {Circle_Diam_1}")
    logger.info(f"entropy_threshold_at: {entropy_threshold_at}")
    logger.info(f"adv_trainer: {adv_trainer}")
    logger.info(f"cifar_weight_decay: {cifar_wd}")
    logger.info(f"mnist_weight_decay: {mnist_wd}")
    logger.info(f"adv_trainer: {adv_trainer}")
    logger.info("****************************")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if chose_dataset == 'cifar10':
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = resnet(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        # model_tmp = models.resnet101(pretrained=True)

        # if not os.path.exists(log_path + "{}/{}/{}/resnet110-1d1ed7c2.th".format(chose_dataset, chose_model, model_layer)):
        #     out_path = log_path + "{}/{}/{}/".format(chose_dataset, chose_model, model_layer)
        #     os.system(f"wget {model_urls['resnet110']} -P {out_path}")
        # chkp1 = torch.load(log_path + "{}/{}/{}/resnet110-1d1ed7c2.th".
        #                    format(chose_dataset, chose_model, model_layer), map_location=device )
        # model.load_state_dict(chkp1, strict=False)

        # if os.path.exists(log_path + "{}/{}/{}/original_best.pt".format(chose_dataset, chose_model, model_layer)):
        #     chkp1 = torch.load(log_path + "{}/{}/{}/original_best.pt".
        #                    format(chose_dataset, chose_model, model_layer), map_location=device )
        #     model.load_state_dict(chkp1, strict=False)
        # else:
        #     out_path = log_path + "{}/{}/{}/".format(chose_dataset, chose_model, model_layer)
        #     os.system(f"wget {model_urls['resnet110']} -P {out_path}")
        #     chkp1 = torch.load(log_path + "{}/{}/{}/resnet110-1d1ed7c2.th".
        #                        format(chose_dataset, chose_model, model_layer), map_location=device )
        #     model.load_state_dict(chkp1, strict=False)

        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, OUTPUT_DIM, bias=True)
        # train_iterator, test_iterator, adv_iterator, nos_train, nos_test = cifar10_loaders(logger, batch_size=cifar_batch)
        # model = Net(OUTPUT_DIM, INPUT_DIM, model_layer)
        # model = resnet(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
    if chose_dataset == 'cifar100':
        # model = Net(OUTPUT_DIM, INPUT_DIM, model_layer)
        model = resnet(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        # if not os.path.exists(log_path + "{}/{}/{}/resnet110-1d1ed7c2.th".format(chose_dataset, chose_model, model_layer)):
        #     out_path = log_path + "{}/{}/{}/".format(chose_dataset, chose_model, model_layer)
        #     os.system(f"wget {model_urls['resnet110']} -P {out_path}")
        # chkp1 = torch.load(log_path + "{}/{}/{}/resnet110-1d1ed7c2.th".
        #                    format(chose_dataset, chose_model, model_layer), map_location=device )
        # model.load_state_dict(chkp1, strict=False)

    if chose_dataset == 'svhn':
        # model = Net(OUTPUT_DIM, INPUT_DIM, model_layer)
        model = resnet(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        # if not os.path.exists(log_path + "{}/{}/{}/resnet110-1d1ed7c2.th".format(chose_dataset, chose_model, model_layer)):
        #     out_path = log_path + "{}/{}/{}/".format(chose_dataset, chose_model, model_layer)
        #     os.system(f"wget {model_urls['resnet110']} -P {out_path}")

        # if os.path.exists(log_path + "{}/{}/{}/original_best.pt".format(chose_dataset, chose_model, model_layer)):
        #     chkp1 = torch.load(log_path + "{}/{}/{}/original_best.pt".
        #                    format(chose_dataset, chose_model, model_layer), map_location=device )
        #     model.load_state_dict(chkp1, strict=False)
        # else:
        #     out_path = log_path + "{}/{}/{}/".format(chose_dataset, chose_model, model_layer)
        #     os.system(f"wget {model_urls['resnet110']} -P {out_path}")
        #     chkp1 = torch.load(log_path + "{}/{}/{}/resnet110-1d1ed7c2.th".
        #                        format(chose_dataset, chose_model, model_layer), map_location=device )
        #     model.load_state_dict(chkp1, strict=False)

    if chose_dataset == 'mnist':
        model = LeNet(OUTPUT_DIM, INPUT_DIM, model_layer)
    if chose_dataset == 'fmnist':
        model = LeNet(OUTPUT_DIM, INPUT_DIM, model_layer)


    if os.path.exists(log_path + "{}/{}/{}/original_best.pt".format(chose_dataset, chose_model, model_layer)):
        chkp1 = torch.load(log_path + "{}/{}/{}/original_best.pt".
                       format(chose_dataset, chose_model, model_layer), map_location=device )
        model.load_state_dict(chkp1, strict=False)

    model = model.to(device)


    if (chose_dataset == 'cifar10') or (chose_dataset == 'cifar100') or (chose_dataset == 'svhn'):
        # optimizer = optim.Adam(model.parameters(), lr=cifar_lr)
        optimizer = optim.SGD(model.parameters(), lr=cifar_lr,
                              momentum=0.9, weight_decay=cifar_wd)
        scheduler = MultiStepLR(optimizer, milestones=[break_point_1, break_point_2], gamma=0.1)
    else:
        optimizer = optim.Adam(model.parameters(), lr=mnist_lr, weight_decay=mnist_wd)
        scheduler = MultiStepLR(optimizer, milestones=[break_point_1], gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # logger = set_logger()

    if chose_dataset == 'cifar10':
        train_iterator, test_iterator,_,_,_ = cifar10_loader(logger, batch_size=cifar_batch)
    if chose_dataset == 'cifar100':
        train_iterator, test_iterator,_,_,_ = cifar100_loader(logger, batch_size=cifar_batch)
    if chose_dataset == 'svhn':
        train_iterator, test_iterator, _, _, _ = svhn_loaders(logger, batch_size=svhn_batch)
    if chose_dataset == 'mnist':
        train_iterator, test_iterator,_,_,_ = mnist_loaders(logger, batch_size=mnist_batch)
    if chose_dataset == 'fmnist':
        train_iterator, test_iterator, _, _, _ = fmnist_loaders(logger, batch_size=fmnist_batch)


    if not os.path.exists(log_path + "{}/{}/{}/original_best.pt".format(chose_dataset, chose_model, model_layer)):
        logger.info("training the original model")
        metric_before_gravity = train_loop(model=model,
                                           logger=logger,
                                           device=device,
                                           optimizer=optimizer,
                                           scheduler=scheduler,
                                           criterion=criterion,
                                           train_iterator=train_iterator,
                                           test_iterator=test_iterator,
                                           chk_fname = 'original_best',
                                           score_fname = "original_score",
                                           black_box = False)
    else:
        logger.info("model is already trained")
        with open(log_path + "{}/{}/{}/original_score.json".format(chose_dataset, chose_model, model_layer)) as p:
            metric_before_gravity = json.load(p)

    #DBSCN HYPER Parameteres:
    dbs_eps_i = DBSCAN_EPS_i
    dbs_smp_i = DBSCAN_SAMPLES_i
    diam_i = Circle_Diam_i

    dbs_eps_1 = DBSCAN_EPS_1
    dbs_smp_1 = DBSCAN_SAMPLES_1
    diam_1 = Circle_Diam_1



    logger.info("extracting the original model latent spaces ...")
    ls_original, _, _ = extract_centroid(model=model,
                                         loader=train_iterator,
                                         device=device,
                                         snapshot_name='original_best',
                                         ls_name='ls_original',
                                         centroid_name='original_centroids',
                                         ls_1_name='ls_1_original',
                                         centroid_1_name='original_1_centroids',
                                         dbs_eps_1=dbs_eps_1,
                                         dbs_smp_1=dbs_smp_1,
                                         dbs_eps_i=dbs_eps_i,
                                         dbs_smp_i=dbs_smp_i)

    logger.info("visualizing the original latent spaces ...")
    vis(ls_name='ls_original',
        component=2,
        technique='pca',
        path=1,
        maped=None)

    vis(ls_name='ls_1_original',
        component=2,
        technique='pca',
        path=1,
        maped=None)
    # vis(ls_name=f'ls_original',
    #     component=2,
    #     technique='tsne')

    # vis(ls_name=f'ls_1_original',
    #     component=2,
    #     technique='tsne')

    logger.info("extracing the distance metrics of the original latent spaces ...")
    # dist_before_gravity = distance_metric(ls_original)
    min_before_gravity, dist_before_gravity = distancetree_metric('original_centroids')
    min_1_before_gravity, dist_1_before_gravity = distancetree_metric('original_1_centroids')
    # logger.info('distance before gravity {}'.format(dist_before_gravity))

    logger.info("creating the modifed centroid and   track of the original model...")
    modified_centroid(centroid_path='original_centroids',
                      ls_path='ls_original',
                      itr=1,
                      coef=1,
                      track_fname='centroid_tracks.csv',
                      mdfy_cnt_fname='modified_centroids',
                      round=0,
                      alfa=1.0,
                      epsilon=0.3,
                      mu=1.0,
                      dbs_eps=dbs_eps_i,
                      dbs_smp=dbs_smp_i,
                      diam=diam_i)


    modified_centroid(centroid_path='original_1_centroids',
                      ls_path='ls_1_original',
                      itr=1,
                      coef=1,
                      track_fname='centroid_1_tracks.csv',
                      mdfy_cnt_fname='modified_1_centroids',
                      round=0,
                      alfa=1.0,
                      epsilon=0.3,
                      mu=1.0,
                      dbs_eps=dbs_eps_1,
                      dbs_smp=dbs_smp_1,
                      diam=diam_1)

    logger.info("start to train the student model...")
    distances = []
    # for alfa in np.linspace(0, 5, 11):
    # for alfa in [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]:
    alfa = metric_before_gravity['test_acc']
    # epsilons_ = np.linspace(0.5, 0.2, CENT_SWITCH)
    epsilons_ = np.array([0.5]*CENT_SWITCH)
    for index in range(CENT_SWITCH):
        logger.info(f"{index}'th-round of centroid chanted...")
        alfa = round(alfa, 5)
        # epsilon = epsilons_[index]
        # mu = 0.5
        temp = {}
        # model, logger, device, optimizer, criterion, train_iterator, test_iterator, alfa = 0.1, coef = 2
        logger.info(f"start train_kd_loop at {index}/{CENT_SWITCH}'th-round of centroid chanted...")
        if adv_trainer == 'fgsm':
            # train_kd_loop(model, logger, device, optimizer, criterion,
            #               train_iterator, test_iterator, index, alfa=0.1,
            #               coef=2, epochs=CENT_SWITCH, epsilon=0.3, mu=1.0, adv_trainer=None)
            metric_after_gravity = train_kd_loop(model=model,
                                                 logger=logger,
                                                 device=device,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 criterion=criterion,
                                                 train_iterator=train_iterator,
                                                 test_iterator=test_iterator,
                                                 index=index,
                                                 alfa=alfa,
                                                 coef=200,
                                                 epochs=EPOCHS,
                                                 epsilon_1=EPSILON_1,
                                                 mu_1=MU_1,
                                                 epsilon_i=EPSILON_i,
                                                 mu_i=MU_i,
                                                 dbs_eps_1=dbs_eps_1,
                                                 dbs_smp_1=dbs_smp_1,
                                                 diam_1=diam_1,
                                                 dbs_eps_i=dbs_eps_i,
                                                 dbs_smp_i=dbs_smp_i,
                                                 diam_i=diam_i,
                                                 adv_trainer='fgsm')

        if adv_trainer == 'pgd':
            metric_after_gravity = train_kd_loop(model=model,
                                                 logger=logger,
                                                 device=device,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 criterion=criterion,
                                                 train_iterator=train_iterator,
                                                 test_iterator=test_iterator,
                                                 index=index,
                                                 alfa=alfa,
                                                 coef=200,
                                                 epochs=EPOCHS,
                                                 epsilon_1=EPSILON_1,
                                                 mu_1=MU_1,
                                                 epsilon_i=EPSILON_i,
                                                 mu_i=MU_i,
                                                 diam_1=diam_1,
                                                 dbs_eps_1=dbs_eps_1,
                                                 dbs_smp_1=dbs_smp_1,
                                                 diam_i=diam_i,
                                                 dbs_eps_i=dbs_eps_i,
                                                 dbs_smp_i=dbs_smp_i,
                                                 adv_trainer='pgd')

        if adv_trainer == None:
            metric_after_gravity = train_kd_loop(model=model,
                                                 logger=logger,
                                                 device=device,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 criterion=criterion,
                                                 train_iterator=train_iterator,
                                                 test_iterator=test_iterator,
                                                 index=index,
                                                 alfa=alfa,
                                                 coef=200,
                                                 epochs=EPOCHS,
                                                 epsilon_1=EPSILON_1,
                                                 mu_1=MU_1,
                                                 epsilon_i=EPSILON_i,
                                                 mu_i=MU_i,
                                                 dbs_eps_1=dbs_eps_1,
                                                 dbs_smp_1=dbs_smp_1,
                                                 diam_1=diam_1,
                                                 dbs_eps_i=dbs_eps_i,
                                                 dbs_smp_i=dbs_smp_i,
                                                 diam_i=diam_i,
                                                 adv_trainer=None)
        logger.info(f"extracting centroids at {index}/{CENT_SWITCH}'th-round of centroid changed...")
        ls_gravity, _, _ = extract_centroid(model=model,
                                            loader=train_iterator,
                                            device=device,
                                            snapshot_name=f'gravity_best_{alfa}_{index}',
                                            ls_name=f'ls_gravity_{alfa}_{index}',
                                            centroid_name=f'gravity_centroids_{alfa}_{index}',
                                            ls_1_name=f'ls_1_gravity_{alfa}_{index}',
                                            centroid_1_name=f'gravity_1_centroids_{alfa}_{index}',
                                            dbs_eps_1=dbs_eps_1,
                                            dbs_smp_1=dbs_smp_1,
                                            dbs_eps_i=dbs_eps_i,
                                            dbs_smp_i=dbs_smp_i)

        vis(ls_name=f'ls_gravity_{alfa}_{index}',
            component=2,
            technique='pca',
            path=1,
            maped=None)

        vis(ls_name=f'ls_1_gravity_{alfa}_{index}',
            component=2,
            technique='pca',
            path=1,
            maped=None)

        logger.info(f"visulizing latent spaces of student at {index}/{CENT_SWITCH}'th-round of centroid changed...")
        # vis(ls_name=f'ls_gravity_{alfa}_{index}', component=2, technique='tsne')
        # vis(ls_name=f'ls_1_gravity_{alfa}_{index}', component=2, technique='tsne')


        temp['index'] = index
        temp['alfa'] = alfa
        temp['min'], temp['dist'] = distancetree_metric(f'gravity_centroids_{alfa}_{index}')
        temp['min_1'], temp['dist_1'] = distancetree_metric(f'gravity_1_centroids_{alfa}_{index}')
        temp['epoch'] = metric_after_gravity['epoch']
        temp['test_acc'] = metric_after_gravity['test_acc']
        temp['test_loss'] = metric_after_gravity['test_loss']
        temp['train_acc'] = metric_after_gravity['train_acc']
        temp['train_loss'] = metric_after_gravity['train_loss']

        distances.append(temp.copy())

        alfa = metric_after_gravity['test_acc']
        # logger.info('distance after gravity  dist {}'.format(distance_metric(ls_gravity)))

    
    temp = {}
    temp['index'] = -1
    temp['alfa'] = np.nan
    temp['dist'] = dist_before_gravity
    temp['min'] = min_before_gravity
    temp['dist_1'] = dist_1_before_gravity
    temp['min_1'] = min_1_before_gravity

    temp['epoch'] = metric_before_gravity['epoch']
    temp['test_acc'] = metric_before_gravity['test_acc']
    temp['test_loss'] = metric_before_gravity['test_loss']
    temp['train_acc'] = metric_before_gravity['train_acc']
    temp['train_loss'] = metric_before_gravity['train_loss']

    logger.info(f"creating the distances.csv")
    distances.append(temp.copy())
                                       
    df = pd.DataFrame(distances)
    df.to_csv(log_path + "{}/{}/{}/distances.csv".
              format(chose_dataset, chose_model, model_layer))

def best_alfa(name):
    df1 = pd.read_csv(log_path + "{}/{}/{}/{}".
                       format(chose_dataset, chose_model, model_layer, name))
    df1.dropna(inplace=True)
    all_test_acc = df1['test_acc'] / np.max(df1['test_acc'])
    all_min_1 = df1['min_1'] / np.max(df1['min_1'])
    all_min_i = df1['min'] / np.max(df1['min'])
    altogether = np.exp(100*all_test_acc) * all_min_1 * all_min_i
    # altogether = all_test_acc
    selected_alfa = df1.loc[np.argmax(altogether)]['alfa']
    selected_alfa = round(selected_alfa, 5)
    selected_index = df1.loc[np.argmax(altogether)]['index']
    logger.info('best_alfa_{}_{}'.format(selected_alfa, int(selected_index)))
    return '{}_{}'.format(selected_alfa, int(selected_index))

###fgsm
os.environ['TORCH_HOME'] = log_path + "{}/{}/{}/".format(chose_dataset, chose_model, model_layer)

###black box attack
def bb_attacks(alfa):
    global logger

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # logger = set_logger()
    if chose_dataset == 'cifar10':

        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        M_O = models.vgg19(pretrained=True)
        num_ftrs = M_O.classifier[6].in_features
        M_O.classifier[6] = nn.Linear(num_ftrs, OUTPUT_DIM, bias=True)
        M_G = resnet_fb(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        # M_O = Net(OUTPUT_DIM, INPUT_DIM)
        # M_G = Net(OUTPUT_DIM, INPUT_DIM)
        #nos is number of samples
        train_iterator, test_iterator, adv_iterator, nos_train, nos_test = cifar10_loader(logger, batch_size=cifar_batch)
    if chose_dataset == 'cifar100':

        preprocessing = dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        M_O = models.vgg19(pretrained=True)
        num_ftrs = M_O.classifier[6].in_features
        M_O.classifier[6] = nn.Linear(num_ftrs, OUTPUT_DIM, bias=True)
        M_G = resnet(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        # M_O = Net(OUTPUT_DIM, INPUT_DIM)
        # M_G = Net(OUTPUT_DIM, INPUT_DIM)
        #nos is number of samples
        train_iterator, test_iterator, adv_iterator, nos_train, nos_test = cifar100_loader(logger, batch_size=cifar_batch)
    if chose_dataset == 'svhn':
        preprocessing = dict(mean=[0.4378, 0.4439, 0.4729], std=[0.1980, 0.2011, 0.1971])
        M_O = models.vgg19(pretrained=True)
        num_ftrs = M_O.classifier[6].in_features
        M_O.classifier[6] = nn.Linear(num_ftrs, OUTPUT_DIM, bias=True)
        M_G = resnet_fb(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        # M_O = Net(OUTPUT_DIM, INPUT_DIM)
        # M_G = Net(OUTPUT_DIM, INPUT_DIM)
        #nos is number of samples
        train_iterator, test_iterator, adv_iterator, nos_train, nos_test = svhn_loaders(logger, batch_size=svhn_batch)
    if chose_dataset == 'mnist':
        # mean = [0.1307]
        # std = [0.3081]
        preprocessing = dict(mean=[0.1307], std=[0.3081])
        M_O = MLeNet(OUTPUT_DIM, INPUT_DIM)
        # M_O.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # num_ftrs = M_O.classifier[6].in_features
        # M_O.classifier[6] = nn.Lin ear(num_ftrs, OUTPUT_DIM, bias=True)
        M_G = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator, adv_iterator, nos_train, nos_test = mnist_loaders(logger, batch_size=mnist_batch)
    if chose_dataset == 'fmnist':
        preprocessing = dict(mean=[0.1307], std=[0.3081])
        M_O = MLeNet(OUTPUT_DIM, INPUT_DIM)
        # M_O = models.vgg19(pretrained=True)
        # M_O.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # num_ftrs = M_O.classifier[6].in_features
        # M_O.classifier[6] = nn.Linear(num_ftrs, OUTPUT_DIM, bias=True)
        M_G = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator, adv_iterator, nos_train, nos_test = fmnist_loaders(logger, batch_size=fmnist_batch)

    if (chose_dataset == 'cifar10') or (chose_dataset == 'cifar100') or (chose_dataset == 'svhn'):
        optimizer = optim.Adam(M_O.parameters(), lr=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[break_point_1], gamma=0.1)
    else:
        optimizer = optim.Adam(M_O.parameters(), lr=mnist_lr)
        scheduler = MultiStepLR(optimizer, milestones=[break_point_1], gamma=0.1)

    # optimizer = optim.Adam(M_O.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    M_O = M_O.to(device)
    M_G = M_G.to(device)

    if (chose_dataset == 'cifar10') or (chose_dataset == 'cifar100') or (chose_dataset == 'svhn'):
        if not os.path.exists(log_path + "{}/{}/{}/bb_VGG19_best.pt".format(chose_dataset, chose_model, model_layer)):
            logger.info("training the black box VGG19")
            train_loop(model=M_O,
                       logger=logger,
                       device=device,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       criterion=criterion,
                       train_iterator=train_iterator,
                       test_iterator=test_iterator,
                       chk_fname = 'bb_VGG19_best',
                       score_fname = "bb_VGG19_score",
                       black_box = True)
        else:
            logger.info("the black box VGG19 is already trained")
            chkp1 = torch.load(log_path + "{}/{}/{}/bb_VGG19_best.pt".
                               format(chose_dataset, chose_model, model_layer), map_location=device, )
            M_O.load_state_dict(chkp1['state_dict'])
    else:
        if not os.path.exists(log_path + "{}/{}/{}/bb_MLeNet_best.pt".format(chose_dataset, chose_model, model_layer)):
            logger.info("training the black box MLeNet")
            train_loop(model=M_O,
                       logger=logger,
                       device=device,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       criterion=criterion,
                       train_iterator=train_iterator,
                       test_iterator=test_iterator,
                       chk_fname = 'bb_MLeNet_best',
                       score_fname = "bb_MLeNet_score",
                       black_box = True)
        else:
            logger.info("the black box MLeNet is already trained")
            chkp1 = torch.load(log_path + "{}/{}/{}/bb_MLeNet_best.pt".
                               format(chose_dataset, chose_model, model_layer), map_location=device, )
            M_O.load_state_dict(chkp1['state_dict'])


    chkp2 = torch.load(log_path + "{}/{}/{}/gravity_best_{}.pt".
                                   format(chose_dataset, chose_model, model_layer, alfa), map_location=device)
    M_G.load_state_dict(chkp2['state_dict'])

    M_O.eval()
    M_G.eval()

    # for atk_name in ['cw']:
    for atk_name in ['fgsm', 'pgd', 'bim', 'mim']:
        logger.info(f"start bb_attack_{atk_name} at {alfa}")
        if atk_name == 'fgsm':
            # acc_eps = {0.05: {}, 0.1: {}, 0.3: {}, 0.7: {}, 1: {}}
            acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
            # if (chose_dataset == 'cifar10') or (chose_dataset == 'svhn') or (chose_dataset == 'cifar100'):
            #     acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05]
            #     # acc_eps = [0.03]
            # else:
            #     acc_eps = [0.01, 0.05, 0.1, 0.2, 0.3]
                # acc_eps = [0.3]


            acc_itrs = [1]*len(acc_eps)

        if atk_name == 'pgd':
            # acc_itrs = {10: {}, 100: {}}
            # acc_itrs = [10]*5
            acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
            # if (chose_dataset == 'cifar10') or (chose_dataset == 'svhn') or (chose_dataset == 'cifar100'):
            #     acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05]
            # else:
            #     acc_eps = [0.01, 0.05, 0.1, 0.2, 0.3]
            acc_itrs = [10] * len(acc_eps)
        if atk_name == 'bim':
            # acc_eps = {0.1: {}, 0.3: {}, 0.7: {}}
            acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
            # if (chose_dataset == 'cifar10') or (chose_dataset == 'svhn') or (chose_dataset == 'cifar100'):
            #     acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05]
            # else:
            #     acc_eps = [0.01, 0.05, 0.1, 0.2, 0.3]

            acc_itrs = [10] * len(acc_eps)

        if atk_name == 'mim':
            # acc_itrs = {10: {}, 50: {}, 100: {}}
            # acc_itrs = [10]*5
            acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
            # if (chose_dataset == 'cifar10') or (chose_dataset == 'svhn') or (chose_dataset == 'cifar100'):
            #     acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05]
            # else:
            #     acc_eps = [0.01, 0.05, 0.1, 0.2, 0.3]
            acc_itrs = [10] * len(acc_eps)

        if atk_name == 'cw':
            # acc_itrs = {10: {}, 50: {}, 100: {}}
            acc_eps = [10, 1, 0.1, 0.01]
            acc_itrs = [1000] * len(acc_eps)


        foolrates = defaultdict(list)
        for acc_itr, eps in zip(acc_itrs, acc_eps):
            M_O_adv_acc = 0
            M_O_clean_acc = 0
            M_G_adv_acc = 0
            M_G_clean_acc = 0

            M_O_avg_psnr = 0
            M_G_avg_psnr = 0

            M_O_avg_mse = 0
            M_G_avg_mse = 0

            M_O_avg_ssim = 0
            M_G_avg_ssim = 0

            M_O_adv_pred = np.array([], dtype=np.int)
            M_O_clean_pred = np.array([], dtype=np.int)
            M_G_adv_pred = np.array([], dtype=np.int)
            M_G_clean_pred = np.array([], dtype=np.int)
            gt = np.array([], dtype=np.int)

            if atk_name == 'fgsm':
                attack_M_O = FGSM(predict=M_O,
                                  loss_fn=criterion,
                                  eps=eps,
                                  clip_min=0.0,
                                  clip_max=1.0,
                                  targeted=False)

            if atk_name == 'pgd':
                attack_M_O = LinfBasicIterativeAttack(predict=M_O,
                                                      loss_fn=criterion,
                                                      eps=eps,
                                                      nb_iter=acc_itr,
                                                      eps_iter=eps / 10,
                                                      clip_min=0.0,
                                                      clip_max=1.0,
                                                      targeted=False)

            if atk_name == 'bim':
                attack_M_O = LinfBasicIterativeAttack(predict=M_O,
                                                      loss_fn=criterion,
                                                      eps=eps,
                                                      nb_iter=acc_itr,
                                                      eps_iter=eps / 10,
                                                      clip_min=0.0,
                                                      clip_max=1.0,
                                                      targeted=False)

            if atk_name == 'mim':
                attack_M_O = LinfMomentumIterativeAttack(predict=M_O,
                                                         loss_fn=criterion,
                                                         eps=eps,
                                                         nb_iter=acc_itr,
                                                         eps_iter=eps / 10,
                                                         clip_min=0.0,
                                                         clip_max=1.0,
                                                         targeted=False)

            if atk_name == 'cw':
                attack_M_O = CarliniWagnerL2Attack(predict=M_O,
                                                   loss_fn=criterion,
                                                   max_iterations=acc_itr,
                                                   num_classes=OUTPUT_DIM,
                                                   learning_rate=0.01,
                                                   confidence=0.0,
                                                   initial_const=eps,
                                                   clip_min=0.0,
                                                   clip_max=1.0,
                                                   targeted=False)

            # attack_M_O = FGSM(predict=M_O, loss_fn=criterion, eps=eps, clip_min=0.0, clip_max=1.0, targeted=False)
            # attack_M_G = FGSM(predict=M_G, loss_fn=criterion, eps=eps, clip_min=0.0, clip_max=1.0, targeted=False)

            ###attack on the teacher model
            for i, (data_batch, labels_batch) in enumerate(adv_iterator):
                gt = np.append(gt, labels_batch.cpu().numpy())
                data_batch = data_batch.to(device)
                labels_batch = labels_batch.to(device)
                ##attack on original model
                output = M_O(normalize(t=data_batch.clone(),
                                       mean=preprocessing['mean'],
                                       std=preprocessing['std'],
                                       dataset=chose_dataset))
                pred_cln = torch.argmax(output, dim=-1)
                M_O_clean_pred = np.append(M_O_clean_pred, pred_cln.cpu().numpy())
                M_O_clean_acc += pred_cln.eq(labels_batch.view_as(pred_cln)).sum().item()

                adv = attack_M_O.perturb(data_batch, labels_batch)
                output = M_O(normalize(t=adv.clone(),
                                       mean=preprocessing['mean'],
                                       std=preprocessing['std'],
                                       dataset=chose_dataset))

                ##calculating PSNR
                temp_mse = nn.MSELoss()(adv, data_batch)
                if np.isclose(temp_mse.item(), 0.0) == False:
                    temp_psnr = 10 * log10(1 / temp_mse.item())
                    temp_ssim = (pytorch_ssim.SSIM())(adv, data_batch)
                    M_O_avg_psnr += temp_psnr
                    M_O_avg_mse += temp_mse.item()
                    M_O_avg_ssim += temp_ssim.item()
                if i == 0:
                    torchvision.utils.save_image(data_batch[0:9],
                                                 log_path + "{}/{}/{}/M_O_clean_bb_{}_{}.png".
                                                            format(chose_dataset, chose_model, model_layer,atk_name, eps),
                                                 nrow=3)

                pred_adv = torch.argmax(output, dim=-1)
                M_O_adv_pred = np.append(M_O_adv_pred, pred_adv.cpu().numpy())
                M_O_adv_acc += pred_adv.eq(labels_batch.view_as(pred_adv)).sum().item()

                ##attack on robusted model
                output = M_G(normalize(t=data_batch.clone(),
                                       mean=preprocessing['mean'],
                                       std=preprocessing['std'],
                                       dataset=chose_dataset))

                pred_cln = torch.argmax(output, dim=-1)
                M_G_clean_pred = np.append(M_G_clean_pred, pred_cln.cpu().numpy())
                # M_O_clean_acc += np.sum(torch.argmax(pred_cln, dim=-1).cpu().numpy() == labels_batch.cpu().numpy())

                M_G_clean_acc += pred_cln.eq(labels_batch.view_as(pred_cln)).sum().item()
                output = M_G(normalize(t=adv.clone(),
                                       mean=preprocessing['mean'],
                                       std=preprocessing['std'],
                                       dataset=chose_dataset))

                ##calculating PSNR
                temp_mse = nn.MSELoss()(adv, data_batch)
                temp_psnr = 10 * log10(1 / temp_mse.item())
                temp_ssim = (pytorch_ssim.SSIM())(adv, data_batch)
                M_G_avg_psnr += temp_psnr
                M_G_avg_mse += temp_mse.item()
                M_G_avg_ssim += temp_ssim.item()
                if i == 0:
                    torchvision.utils.save_image(adv[0:9],
                                                 log_path + "{}/{}/{}/M_G_adv_bb_{}_{}.png".
                                                            format(chose_dataset, chose_model, model_layer, atk_name, eps),
                                                 nrow=3)

                pred_adv = torch.argmax(output, dim=-1)
                M_G_adv_pred = np.append(M_G_adv_pred, pred_adv.cpu().numpy())
                M_G_adv_acc += pred_adv.eq(labels_batch.view_as(pred_adv)).sum().item()

                logger.info(f'bb {atk_name} epsilon {eps} itr {acc_itr} Batch: {i}')

            foolrates[acc_itr] = {'MO_clean_acc': M_O_clean_acc / nos_test,
                                  'MO_adv_acc': M_O_adv_acc / nos_test,
                                  'MO_PSNR' : M_O_avg_psnr/len(adv_iterator),
                                  'MO_MSE': M_O_avg_mse / len(adv_iterator),
                                  'MO_SSIM': M_O_avg_ssim / len(adv_iterator),
                                  'MG_clean_acc': M_G_clean_acc / nos_test,
                                  'MG_adv_acc': M_G_adv_acc / nos_test,
                                  'MG_PSNR' : M_G_avg_psnr/len(adv_iterator),
                                  'MG_MSE' : M_G_avg_mse/len(adv_iterator),
                                  'MG_SSIM': M_G_avg_ssim / len(adv_iterator)}.copy()

            foolrates[eps] = {'MO_clean_acc': M_O_clean_acc / nos_test,
                              'MO_adv_acc': M_O_adv_acc / nos_test,
                              'MO_PSNR': M_O_avg_psnr / len(adv_iterator),
                              'MO_MSE': M_O_avg_mse / len(adv_iterator),
                              'MO_SSIM': M_O_avg_ssim / len(adv_iterator),
                              'MG_clean_acc': M_G_clean_acc / nos_test,
                              'MG_adv_acc': M_G_adv_acc / nos_test,
                              'MG_PSNR' : M_G_avg_psnr/len(adv_iterator),
                              'MG_MSE' : M_G_avg_mse/len(adv_iterator),
                              'MG_SSIM': M_G_avg_ssim / len(adv_iterator)}.copy()

            ## this section save the confusion matrix for each one of the attacks
            conf_mtx = confusion_matrix(gt, M_O_clean_pred)
            joblib.dump(conf_mtx, log_path + "{}/{}/{}/M_O_clean_bb_{}_{}.mtx".
                        format(chose_dataset, chose_model, model_layer, atk_name, eps))


            conf_mtx = confusion_matrix(gt, M_O_adv_pred)
            joblib.dump(conf_mtx, log_path + "{}/{}/{}/M_O_adv_bb_{}_{}.mtx".
                        format(chose_dataset, chose_model, model_layer, atk_name, eps))

            # with open(log_path + "{}/{}/{}/M_O_adv_bb_{}.mtx".format(chose_dataset, chose_model, model_layer, atk_name),
            #           "w") as p:
            #     json.dump(conf_mtx, p)

            conf_mtx = confusion_matrix(gt, M_G_clean_pred)
            joblib.dump(conf_mtx, log_path + "{}/{}/{}/M_G_clean_bb_{}_{}.mtx".
                        format(chose_dataset, chose_model, model_layer, atk_name, eps))
            # with open(log_path + "{}/{}/{}/M_G_clean_bb.mtx".format(chose_dataset, chose_model, model_layer),
            #           "w") as p:
            #     json.dump(conf_mtx, p)

            conf_mtx = confusion_matrix(gt, M_G_adv_pred)
            joblib.dump(conf_mtx, log_path + "{}/{}/{}/M_G_adv_bb_{}_{}.mtx".
                        format(chose_dataset, chose_model, model_layer, atk_name, eps))
            # with open(log_path + "{}/{}/{}/M_G_adv_bb_{}.mtx".format(chose_dataset, chose_model, model_layer, atk_name),
            #           "w") as p:
            #     json.dump(conf_mtx, p)
    

        with open(log_path + "{}/{}/{}/bb_{}_attack.json".format(
                chose_dataset, chose_model, model_layer, atk_name),"w") as p:
            json.dump(foolrates, p)

        logger.info(f"finish bb_attack_{atk_name} at {alfa}")

###white box attack
def wb_attacks_test(alfa):
    global logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logger = set_logger()
    if chose_dataset == 'cifar10':
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        M_O = resnet_fb(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        M_G = resnet_fb(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)

        # M_O = Net(OUTPUT_DIM, INPUT_DIM)
        # M_G = Net(OUTPUT_DIM, INPUT_DIM)
        #nos is number of samples
        train_iterator, test_iterator,adv_iterator, nos_train, nos_test = cifar10_loader(logger, batch_size=cifar_batch)
    if chose_dataset == 'cifar100':
        mean = [0.507, 0.487, 0.441]
        std = [0.267, 0.256, 0.276]
        M_O = resnet(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        M_G = resnet(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        # M_O = Net(OUTPUT_DIM, INPUT_DIM)
        # M_G = Net(OUTPUT_DIM, INPUT_DIM)
        #nos is number of samples
        train_iterator, test_iterator,adv_iterator, nos_train, nos_test = cifar100_loader(logger, batch_size=cifar_batch)
    if chose_dataset == 'svhn':
        preprocessing = dict(mean=[0.4378, 0.4439, 0.4729], std=[0.1980, 0.2011, 0.1971])
        M_O = resnet_fb(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        M_G = resnet_fb(num_classes=OUTPUT_DIM, depth=110, layer=model_layer)
        # M_O = Net(OUTPUT_DIM, INPUT_DIM)
        # M_G = Net(OUTPUT_DIM, INPUT_DIM)
        #nos is number of samples
        train_iterator, test_iterator,adv_iterator, nos_train, nos_test = svhn_loaders(logger, batch_size=svhn_batch)
    if chose_dataset == 'mnist':
        # mean = [0.1307]
        # std = [0.3081]
        preprocessing = dict(mean=[0.1307], std=[0.3081])
        # mean = torch.tensor(mean, requires_grad=False, device=device)
        # std = torch.tensor(std, requires_grad=False, device=device)
        M_O = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        M_G = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator,adv_iterator, nos_train, nos_test = mnist_loaders(logger, batch_size=mnist_batch)
    if chose_dataset == 'fmnist':
        # mean = [0.1307]
        # std = [0.3081]
        preprocessing = dict(mean=[0.1307], std=[0.3081])
        M_O = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        M_G = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator,adv_iterator, nos_train, nos_test = fmnist_loaders(logger, batch_size=fmnist_batch)

    chkp1 = torch.load(log_path + "{}/{}/{}/original_best.pt".
                                   format(chose_dataset, chose_model, model_layer), map_location=device)
    M_O.load_state_dict(chkp1['state_dict'])

    chkp2 = torch.load(log_path + "{}/{}/{}/gravity_best_{}.pt".
                                   format(chose_dataset, chose_model, model_layer, alfa), map_location=device)
    M_G.load_state_dict(chkp2['state_dict'])

    M_O = M_O.to(device)
    M_G = M_G.to(device)
    M_O.eval()
    M_G.eval()

    # if (chose_dataset == 'cifar10') or (chose_dataset == 'cifar100') or (chose_dataset == 'svhn'):
    #     optimizer = optim.Adam(model.parameters(), lr=cifar_lr)
    # else:
    #     optimizer = optim.Adam(model.parameters(), lr=mnist_lr)


    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # for atk_name in ['cw']:
    for atk_name in ['fgsm', 'pgd', 'bim', 'mim']:
        logger.info(f"start wb_attack_{atk_name} at {alfa}")
        if atk_name == 'fgsm':
            # acc_eps = {0.05: {}, 0.1: {}, 0.3: {}, 0.7: {}, 1: {}}
            acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
            # if (chose_dataset == 'cifar10') or (chose_dataset == 'svhn') or (chose_dataset == 'cifar100'):
            #     acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05]
            #     # acc_eps = [0.03]
            # else:
            #     acc_eps = [0.01, 0.05, 0.1, 0.2, 0.3]
                # acc_eps = [0.3]
            acc_itrs = [1]*len(acc_eps)

        if atk_name == 'pgd':
            # acc_itrs = {10: {}, 100: {}}
            acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
            # if (chose_dataset == 'cifar10') or (chose_dataset == 'svhn') or (chose_dataset == 'cifar100'):
            #     acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05]
            # else:
            #     acc_eps = [0.01, 0.05, 0.1, 0.2, 0.3]

            acc_itrs = [10]*len(acc_eps)

        if atk_name == 'bim':
            # acc_eps = {0.1: {}, 0.3: {}, 0.7: {}}
            acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
            # if (chose_dataset == 'cifar10') or (chose_dataset == 'svhn') or (chose_dataset == 'cifar100'):
            #     acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05]
            # else:
            #     acc_eps = [0.01, 0.05, 0.1, 0.2, 0.3]

            acc_itrs = [10] * len(acc_eps)

        if atk_name == 'mim':
            # acc_itrs = {10: {}, 50: {}, 100: {}}
            # acc_itrs = [10]
            acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
            # if (chose_dataset == 'cifar10') or (chose_dataset == 'svhn') or (chose_dataset == 'cifar100'):
            #     acc_eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
            # else:
            #     acc_eps = [0.01, 0.05, 0.1, 0.2, 0.3]

            acc_itrs = [10] * len(acc_eps)

        if atk_name == 'cw':
            # acc_itrs = {10: {}, 50: {}, 100: {}}
            acc_eps = [10, 1, 0.1, 0.01]
            acc_itrs = [1000]* len(acc_eps)


        foolrates = defaultdict(list)
        for acc_itr, eps in zip(acc_itrs, acc_eps):
            M_O_adv_acc = 0
            M_O_clean_acc = 0
            M_G_adv_acc = 0
            M_G_clean_acc = 0

            M_O_avg_psnr = 0
            M_G_avg_psnr = 0

            M_O_avg_mse = 0
            M_G_avg_mse = 0

            M_O_avg_ssim = 0
            M_G_avg_ssim = 0

            M_O_adv_pred = np.array([])
            M_O_clean_pred = np.array([])
            M_G_adv_pred = np.array([])
            M_G_clean_pred = np.array([])
            gt = np.array([])



            if atk_name == 'fgsm':
                attack_M_O = FGSM(predict=M_O,
                                  loss_fn=criterion,
                                  eps=eps,
                                  clip_min=0.0,
                                  clip_max=1.0,
                                  targeted=False)
                attack_M_G = FGSM(predict=M_G,
                                  loss_fn=criterion,
                                  eps=eps,
                                  clip_min=0.0,
                                  clip_max=1.0,
                                  targeted=False)

            if atk_name == 'pgd':
                attack_M_O = LinfPGDAttack(predict=M_O,
                                         loss_fn=criterion,
                                         eps=eps,
                                         nb_iter=acc_itr,
                                         eps_iter=eps / 10,
                                         clip_min=0.0,
                                         clip_max=1.0,
                                         targeted=False)
                attack_M_G = LinfPGDAttack(predict=M_G,
                                         loss_fn=criterion,
                                         eps=eps,
                                         nb_iter=acc_itr,
                                         eps_iter=eps/10,
                                         clip_min=0.0,
                                         clip_max=1.0,
                                         targeted=False)

            if atk_name == 'bim':
                attack_M_O = LinfBasicIterativeAttack(predict=M_O,
                                                    loss_fn=criterion,
                                                    eps=eps,
                                                    nb_iter=acc_itr,
                                                    eps_iter=eps / 10,
                                                    clip_min=0.0,
                                                    clip_max=1.0,
                                                    targeted=False)
                attack_M_G = LinfBasicIterativeAttack(predict=M_G,
                                                    loss_fn=criterion,
                                                    eps=eps,
                                                    nb_iter=acc_itr,
                                                    eps_iter=eps / 10,
                                                    clip_min=0.0,
                                                    clip_max=1.0,
                                                    targeted=False)

            if atk_name == 'mim':
                attack_M_O = LinfMomentumIterativeAttack(predict=M_O,
                                                       loss_fn=criterion,
                                                       eps=eps,
                                                       nb_iter=acc_itr,
                                                       eps_iter=eps / 10,
                                                       clip_min=0.0,
                                                       clip_max=1.0,
                                                       targeted=False)
                attack_M_G = LinfMomentumIterativeAttack(predict=M_G,
                                                       loss_fn=criterion,
                                                       eps=eps,
                                                       nb_iter=acc_itr,
                                                       eps_iter=eps / 10,
                                                       clip_min=0.0,
                                                       clip_max=1.0,
                                                       targeted=False)

            if atk_name == 'cw':
                attack_M_O = CarliniWagnerL2Attack(predict=M_O,
                                                   loss_fn=criterion,
                                                   max_iterations=acc_itr,
                                                   num_classes=OUTPUT_DIM,
                                                   learning_rate=0.01,
                                                   confidence=0.0,
                                                   initial_const=eps,
                                                   clip_min=0.0,
                                                   clip_max=1.0,
                                                   targeted=False)

                attack_M_G = CarliniWagnerL2Attack(predict=M_G,
                                                   loss_fn=criterion,
                                                   max_iterations=acc_itr,
                                                   num_classes=OUTPUT_DIM,
                                                   learning_rate=0.01,
                                                   confidence=0.0,
                                                   initial_const=eps,
                                                   clip_min=0.0,
                                                   clip_max=1.0,
                                                   targeted=False)

            ###attack on the teacher model
            for i, (data_batch, labels_batch) in enumerate(adv_iterator):
                gt = np.append(gt, labels_batch.cpu().numpy())
                data_batch = data_batch.to(device)
                labels_batch = labels_batch.to(device)
                # data_batch = data_batch.to(device)
                # labels_batch = labels_batch.to(device)


                ##attack on original model
                # M_O_clean_acc += torch.sum(
                #     M_O(normalize(data_batch.clone().detach(), mean, std, chose_dataset))[0].argmax(dim=-1) == labels_batch).item()

                # adv = attack_M_O(data_batch, labels_batch, epsilon=eps, iterations=acc_itr)
                output = M_O(normalize(t=data_batch.clone(),
                                       mean=preprocessing['mean'],
                                       std=preprocessing['std'],
                                       dataset=chose_dataset))

                pred_cln = torch.argmax(output, dim=-1)
                M_O_clean_pred = np.append(M_O_clean_pred, pred_cln.cpu().numpy())

                # M_O_clean_acc += np.sum(torch.argmax(pred_cln, dim=-1).cpu().numpy() == labels_batch.cpu().numpy())

                M_O_clean_acc += pred_cln.eq(labels_batch.view_as(pred_cln)).sum().item()

                adv = attack_M_O.perturb(data_batch, labels_batch)
                output = M_O(normalize(t=adv.clone(),
                                       mean=preprocessing['mean'],
                                       std=preprocessing['std'],
                                       dataset=chose_dataset))

                ##calculating PSNR if it was successful attack
                temp_mse = nn.MSELoss()(adv, data_batch)
                if np.isclose(temp_mse.item(), 0.0) == False:
                    temp_psnr = 10 * log10(1 / temp_mse.item())
                    temp_ssim = (pytorch_ssim.SSIM())(adv, data_batch)
                    M_O_avg_psnr += temp_psnr
                    M_O_avg_mse += temp_mse.item()
                    M_O_avg_ssim += temp_ssim.item()
                if i == 0:
                    torchvision.utils.save_image(data_batch[0:9],
                                                 log_path + "{}/{}/{}/M_O_clean_wb_{}.png".
                                                 format(chose_dataset, chose_model, model_layer, atk_name),
                                                 nrow = 3)

                    torchvision.utils.save_image(adv[0:9],
                                                 log_path + "{}/{}/{}/M_O_adv_wb_{}_{}.png".
                                                 format(chose_dataset, chose_model, model_layer, atk_name, eps),
                                                 nrow = 3)

                pred_adv = torch.argmax(output, dim=-1)
                M_O_adv_pred = np.append(M_O_adv_pred, pred_adv.cpu().numpy())


                M_O_adv_acc += pred_adv.eq(labels_batch.view_as(pred_adv)).sum().item()

                ###student section
                output = M_G(normalize(t=data_batch.clone(),
                                       mean=preprocessing['mean'],
                                       std=preprocessing['std'],
                                       dataset=chose_dataset))


                pred_cln = torch.argmax(output, dim=-1)
                M_G_clean_pred = np.append(M_G_clean_pred, pred_cln.cpu().numpy())
                # M_O_clean_acc += np.sum(torch.argmax(pred_cln, dim=-1).cpu().numpy() == labels_batch.cpu().numpy())

                M_G_clean_acc += pred_cln.eq(labels_batch.view_as(pred_cln)).sum().item()

                adv = attack_M_G.perturb(data_batch, labels_batch)
                output = M_G(normalize(t=adv.clone(),
                                       mean=preprocessing['mean'],
                                       std=preprocessing['std'],
                                       dataset=chose_dataset))

                ##calculating PSNR
                temp_mse = nn.MSELoss()(adv, data_batch)
                if np.isclose(temp_mse.item(), 0.0) == False:
                    temp_psnr = 10 * log10(1 / temp_mse.item())
                    temp_ssim = (pytorch_ssim.SSIM())(adv, data_batch)
                    M_G_avg_psnr += temp_psnr
                    M_G_avg_mse += temp_mse.item()
                    M_G_avg_ssim += temp_ssim.item()
                if i == 0:
                    torchvision.utils.save_image(adv[0:9],
                                                 log_path + "{}/{}/{}/M_G_adv_wb_{}_{}.png".
                                                 format(chose_dataset, chose_model, model_layer, atk_name, eps),
                                                 nrow = 3)

                pred_adv = torch.argmax(output, dim=-1)
                M_G_adv_pred = np.append(M_G_adv_pred, pred_adv.cpu().numpy())
                M_G_adv_acc += pred_adv.eq(labels_batch.view_as(pred_adv)).sum().item()

                logger.info(f'wb {atk_name} epsilon {eps} itr {acc_itr} Batch: {i}')

            foolrates[acc_itr] = {'MO_clean_acc': M_O_clean_acc / nos_test,
                                  'MO_adv_acc': M_O_adv_acc / nos_test,
                                  'MO_PSNR' : M_O_avg_psnr/len(adv_iterator),
                                  'MO_MSE': M_O_avg_mse / len(adv_iterator),
                                  'MO_SSIM': M_O_avg_ssim / len(adv_iterator),
                                  'MG_clean_acc': M_G_clean_acc / nos_test,
                                  'MG_adv_acc': M_G_adv_acc / nos_test,
                                  'MG_PSNR' : M_G_avg_psnr/len(adv_iterator),
                                  'MG_MSE' : M_G_avg_mse/len(adv_iterator),
                                  'MG_SSIM': M_G_avg_ssim / len(adv_iterator)}.copy()

            foolrates[eps] = {'MO_clean_acc': M_O_clean_acc / nos_test,
                              'MO_adv_acc': M_O_adv_acc / nos_test,
                              'MO_PSNR': M_O_avg_psnr / len(adv_iterator),
                              'MO_MSE': M_O_avg_mse / len(adv_iterator),
                              'MO_SSIM': M_O_avg_ssim / len(adv_iterator),
                              'MG_clean_acc': M_G_clean_acc / nos_test,
                              'MG_adv_acc': M_G_adv_acc / nos_test,
                              'MG_PSNR' : M_G_avg_psnr/len(adv_iterator),
                              'MG_MSE' : M_G_avg_mse/len(adv_iterator),
                              'MG_SSIM': M_G_avg_ssim / len(adv_iterator)}.copy()
            # foolrates[acc_itr] = {'MO_clean_acc': M_O_clean_acc / len(adv_iterator),
            #                       'MO_adv_acc': M_O_adv_acc / len(adv_iterator),
            #                       'MG_clean_acc': M_G_clean_acc / len(adv_iterator),
            #                       'MG_adv_acc': M_G_adv_acc / len(adv_iterator)}.copy()
            #
            # foolrates[eps] = {'MO_clean_acc': M_O_clean_acc / len(adv_iterator),
            #                   'MO_adv_acc': M_O_adv_acc / len(adv_iterator),
            #                   'MG_clean_acc': M_G_clean_acc / len(adv_iterator),
            #                   'MG_adv_acc': M_G_adv_acc / len(adv_iterator)}.copy()


            ## this section save the confusion matrix for each one of the attacks
            conf_mtx = confusion_matrix(gt, M_O_clean_pred)
            joblib.dump(conf_mtx, log_path + "{}/{}/{}/M_O_clean_wb_{}_{}.mtx".
                        format(chose_dataset, chose_model, model_layer, atk_name, eps))


            # with open(log_path + "{}/{}/{}/M_O_clean_wb.mtx".format(chose_dataset, chose_model, model_layer),
            #           "w") as p:
            #     json.dump(conf_mtx, p)

            conf_mtx = confusion_matrix(gt, M_O_adv_pred)
            joblib.dump(conf_mtx, log_path + "{}/{}/{}/M_O_adv_wb_{}_{}.mtx".
                        format(chose_dataset, chose_model, model_layer, atk_name, eps))

            # with open(log_path + "{}/{}/{}/M_O_adv_wb_{}.mtx".format(chose_dataset, chose_model, model_layer, atk_name),
            #           "w") as p:
            #     json.dump(conf_mtx, p)

            conf_mtx = confusion_matrix(gt, M_G_clean_pred)
            joblib.dump(conf_mtx, log_path + "{}/{}/{}/M_G_clean_wb_{}_{}.mtx".
                        format(chose_dataset, chose_model, model_layer, atk_name, eps))
            # with open(log_path + "{}/{}/{}/M_G_clean_wb.mtx".format(chose_dataset, chose_model, model_layer),
            #           "w") as p:
            #     json.dump(conf_mtx, p)

            conf_mtx = confusion_matrix(gt, M_G_adv_pred)
            joblib.dump(conf_mtx, log_path + "{}/{}/{}/M_G_adv_wb_{}_{}.mtx".
                        format(chose_dataset, chose_model, model_layer, atk_name, eps))

            # with open(log_path + "{}/{}/{}/M_G_adv_wb_{}.mtx".format(chose_dataset, chose_model, model_layer, atk_name),
            #           "w") as p:
            #     json.dump(conf_mtx, p)

            ## this section save the metrix for each one of the attacks
        with open(log_path + "{}/{}/{}/wb_{}_attack_test.json".format(
                chose_dataset, chose_model, model_layer, atk_name),"w") as p:
            json.dump(foolrates, p)

        logger.info(f"finish wb_attack_test{atk_name} at {alfa}")

##extractor
def dbs_extractor(ls_name):
    global logger
    logger.info('extracting the dbscan param')

    ls_original = joblib.load(log_path + "{}/{}/{}/{}".
                              format(chose_dataset, chose_model, model_layer, ls_name))

    ls_original_normal = defaultdict(list)
    maped = defaultdict(list)

    # epss, smps = [0.9], [30]
    epss = np.linspace(5, 20, 10, dtype=np.float16)
    epss = np.append(epss, epss)

    smps = np.linspace(3, 20, 10, dtype=np.int)
    smps = np.append(smps, -1 * np.sort(-smps))

    selected = []
    for eps, smp in zip(epss, smps):
        count = 0
        uniqness = []
        for key, item in ls_original.items():
            #     item_ = StandardScaler().fit_transform(item)
            item_ = MinMaxScaler().fit_transform(item)
            #     item_ = preprocessing.scale(item)
            ls_original_normal[key] = item_.copy()

            dbs_model = DBSCAN(eps=eps, min_samples=smp, metric="euclidean", n_jobs=-1).fit(item_)
            maped[key] = dbs_model.labels_

            #         temp_uniqness = np.unique(dbs_model.labels_)
            if len(np.unique(dbs_model.labels_)) < 3:
                count += 1
                uniqness.append(np.unique(dbs_model.labels_, return_counts=True))
                #         print(key)
        #         print(np.unique(dbs_model.labels_, return_counts=True))
        #         if np.all(dbs_model.labels_ == -1):
        #             print('what the fuck')
        if count == 10:
            selected.append({'eps': eps, 'smp': smp, 'info': uniqness.copy()})

    logger.info(f'{selected}')
    logger.info('exiting the dbscan param')


# main_trainer()

 # wb_original_attacker()
# wb_gravity_attacker()
# bb_attacker()
# selected_alfa = best_alfa(name='distances.csv')
selected_alfa = '0.91837_5'
wb_attacks_test(alfa=selected_alfa)

bb_attacks(alfa=selected_alfa)
# wb_attack_fgsm(alfa=selected_alfa)
# wb_attack_pgd(alfa=selected_alfa)
# wb_attack_bim(alfa=selected_alfa)
# wb_attack_mim(alfa=selected_alfa)
# bb_attack_fgsm(alfa=selected_alfa)
# bb_attack_pgd(alfa=selected_alfa)
# bb_attack_bim(alfa=selected_alfa)
# bb_attack_mim(alfa=selected_alfa)
# wb_attack_cw(alfa=selected_alfa)

# dbs_extractor(ls_name='ls_original')