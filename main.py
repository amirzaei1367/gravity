## imported module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.externals import joblib
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

import numpy as np
import matplotlib.pylab as plt


from model import *
from settings import *
from helpers import *

import foolbox as foolbox
from foolbox import *

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

### setting up the log
def set_logger():
    if not os.path.exists(log_path+"{}".format(chose_dataset)):
        os.mkdir(log_path+"{}".format(chose_dataset))

    if not os.path.exists(log_path+"{}/{}".format(chose_dataset, chose_model)):
        os.mkdir(log_path+"{}/{}".format(chose_dataset, chose_model))

    if not os.path.exists(log_path+"{}/{}/{}".format(chose_dataset, chose_model, model_layer)):
        os.mkdir(log_path+"{}/{}/{}".format(chose_dataset, chose_model, model_layer))

    logging.basicConfig(filename=log_path+"{}/{}/{}/log.log".format(chose_dataset, chose_model, model_layer),
                                format='%(asctime)s %(message)s',
                                filemode='w')
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return  logger

## setting up the dataloaders
def cifar10_loader(logger, batch_size=1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_transformer = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    trainset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transformer)
    devset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=dev_transformer)



    train_iterator = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=True)

    test_iterator = torch.utils.data.DataLoader(devset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=8,
                                            pin_memory=True,
                                            drop_last=True)


    return  train_iterator, test_iterator

def mnist_loaders(logger, batch_size=1):
    train_data = datasets.MNIST(root = dataset_path,
                                train = True,
                                download = False)

    mean = train_data.data.float().mean() / 255
    std = train_data.data.float().std() / 255

    logger.info(f'Calculated mean: {mean}')
    logger.info(f'Calculated std: {std}')

    data_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean],
                             std=[std])
    ])

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

    test_iterator = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return  train_iterator, test_iterator

## training loops
def train_loop(model, logger, device, optimizer, criterion, train_iterator, test_iterator):
    # model = LeNet(OUTPUT_DIM, INPUT_DIM)
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    # optimizer = optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss()
    # # model = model.to(device)
    # criterion = criterion.to(device)

    best_test_loss = float('inf')
    best_test_acc = 0.0

    for epoch in range(EPOCHS):
        # print('idx epoch {}/{} best_loss {}'.format(epoch, EPOCHS, best_test_loss))

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            logger.info("new best at epoch {} with acc {}".format(epoch, test_acc))
            logger.info("new best at epoch {} with acc {}".format(epoch, test_loss))
            torch.save(model.state_dict(), log_path + "{}/{}/{}/original_best.pth".
                       format(chose_dataset, chose_model, model_layer))

            var = {'epoch':epoch,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                   'train_acc': train_acc,
                   'train_loss': train_loss}
            with open(log_path + "{}/{}/{}/original_score.json".format(chose_dataset, chose_model, model_layer), "w") as p:
                json.dump(var, p)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        logger.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        logger.info(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')

    return var

def train_kd_loop(model, logger, device, optimizer, criterion, train_iterator, test_iterator, alfa= 0.1):
    # model = LeNet(OUTPUT_DIM)
    logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    best_test_loss = float('inf')
    for epoch in range(EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train_kd(model, train_iterator, optimizer,
                                         criterion, device, centroid='modified_centroids',
                                         alfa=alfa)
        test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            logger.info("new best at epoch {} with acc {}".format(epoch, test_acc))
            logger.info("new best at epoch {} with acc {}".format(epoch, test_acc))
            torch.save(model.state_dict(), log_path + "{}/{}/{}/gravity_best_{}.pth".
                       format(chose_dataset, chose_model, model_layer, alfa))

            var = {'epoch':epoch,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                   'train_acc': train_acc,
                   'train_loss': train_loss}
            with open(log_path + "{}/{}/{}/gravity_score.json".format(chose_dataset, chose_model, model_layer), "w") as p:
                json.dump(var, p)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        logger.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        logger.info(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')

    return var

def main_trainer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if chose_dataset == 'cifar10':
        model = Net(OUTPUT_DIM, INPUT_DIM, model_layer)
    if chose_dataset == 'mnist':
        model = LeNet(OUTPUT_DIM, INPUT_DIM, model_layer)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    logger=set_logger()

    if not os.path.exists(log_path + "{}/{}/{}/original_best.pth".format(chose_dataset, chose_model, model_layer)):
        if chose_dataset == 'cifar10':
            train_iterator, test_iterator = cifar10_loader(logger, batch_size=64)
        if chose_dataset == 'mnist':
            train_iterator, test_iterator = mnist_loaders(logger, batch_size=64)

        metric_before_gravity = train_loop(model, logger, device, optimizer, criterion, train_iterator, test_iterator)
    else:
        metric_before_gravity = joblib.load(log_path + "{}/{}/{}/original_score.json".format(chose_dataset, chose_model, model_layer))

    if chose_dataset == 'cifar10':
        train_iterator, test_iterator = cifar10_loader(logger, batch_size=1)
    if chose_dataset == 'mnist':
        train_iterator, test_iterator = mnist_loaders(logger, batch_size=1)


    ls_original, _ = extract_centroid(model=model, loader=train_iterator,
                        device=device, snapshot_name='original_best',
                        ls_name='ls_original', centroid_name='original_centroids')
    vis(ls_name=f'ls_original', component=2, technique='tsne')

        # dist_before_gravity = distance_metric(ls_original)
    min_before_gravity, dist_before_gravity = distancetree_metric('original_centroids')
        # logger.info('distance before gravity {}'.format(dist_before_gravity))
    modified_centroid(centroid_path='original_centroids', ls_path='ls_original', itr=4 , coef=1.5)

    distances = []
    # for alfa in np.linspace(0, 5, 11):
    for alfa in [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]:
    # for alfa in [1]:
        temp = {}

        if chose_dataset == 'cifar10':
            train_iterator, test_iterator = cifar10_loader(logger, batch_size=64)
        if chose_dataset == 'mnist':
            train_iterator, test_iterator = mnist_loaders(logger, batch_size=64)

        metric_after_gravity = train_kd_loop(model, logger, device, optimizer,
                                             criterion, train_iterator, test_iterator, alfa=alfa)

        if chose_dataset == 'cifar10':
            train_iterator, test_iterator = cifar10_loader(logger, batch_size=1)
        if chose_dataset == 'mnist':
            train_iterator, test_iterator = mnist_loaders(logger, batch_size=1)

        ls_gravity, _ = extract_centroid(model=model, loader=train_iterator,
                         device=device, snapshot_name=f'gravity_best_{alfa}',
                         ls_name=f'ls_gravity_{alfa}', centroid_name=f'gravity_centroids_{alfa}')

        vis(ls_name=f'ls_gravity_{alfa}', component=2, technique='tsne')

        # temp['alfa'] = alfa
        # temp['dist'] = distance_metric(ls_gravity)


        temp['alfa'] = alfa
        temp['min'], temp['dist'] = distancetree_metric(f'gravity_centroids_{alfa}')
        temp['epoch'] = metric_after_gravity['epoch']
        temp['test_acc'] = metric_after_gravity['test_acc']
        temp['test_loss'] = metric_after_gravity['test_loss']
        temp['train_acc'] = metric_after_gravity['train_acc']
        temp['train_loss'] = metric_after_gravity['train_loss']


        distances.append(temp.copy())
        # logger.info('distance after gravity  dist {}'.format(distance_metric(ls_gravity)))

    temp = {}
    temp['alfa'] = np.nan
    temp['dist'] = dist_before_gravity
    temp['min']  = min_before_gravity

    temp['epoch'] = metric_before_gravity['epoch']
    temp['test_acc'] = metric_before_gravity['test_acc']
    temp['test_loss'] = metric_before_gravity['test_loss']
    temp['train_acc'] = metric_before_gravity['train_acc']
    temp['train_loss'] = metric_before_gravity['train_loss']

    distances.append(temp.copy())

    df = pd.DataFrame(distances)
    df.to_csv(log_path + "{}/{}/{}/distances.csv".
                       format(chose_dataset, chose_model, model_layer))

def wb_original_attacker():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = set_logger()
    if chose_dataset == 'cifar10':
        M_O = Net_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator = cifar10_loader(logger, batch_size=64)
    if chose_dataset == 'mnist':
        M_O = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator = mnist_loaders(logger, batch_size=64)

    M_O = M_O.to(device)
    M_O.eval()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
                                                                                       
    M_O.load_state_dict(torch.load(log_path + "{}/{}/{}/original_best.pth".
                                   format(chose_dataset, chose_model, model_layer)))
    # test_loss, test_acc = evaluate(M_O, test_iterator, criterion, device)
    # print(f'\t M_O Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')

    min_total, max_total = min_max(train_iterator)
    acc_eps = {0.05:{}, 0.1:{}, 0.2:{}, 0.3:{}}

    if chose_dataset == 'cifar10':
        train_iterator, test_iterator = cifar10_loader(logger, batch_size=1)
    if chose_dataset == 'mnist':
        train_iterator, test_iterator = mnist_loaders(logger, batch_size=1)

    FM_O = foolbox.models.PyTorchModel(M_O, bounds=(min_total, max_total), num_classes=10)
    attack_M_O = foolbox.attacks.FGSM(FM_O)

    perturbation_norms  = defaultdict(list)
    foolrates = defaultdict(list)
    for eps, _ in acc_eps.items():
        # teacher_adversarial_images = []
        # student_adversarial_images = []
        foolrate = 0
        robust = 0
        become_adversarial = 0
        already_adversarial = 0

        ###attack on the teacher model
        for i, (data_batch, labels_batch) in enumerate(train_iterator):

            img_numpy = data_batch.cpu().numpy()
            label_numpy = labels_batch.cpu().numpy()

            # print('shape ', img_numpy.shape)
            adversarial = attack_M_O(img_numpy, label_numpy,
                                     unpack=False, epsilons=[eps])

            ##Does this mean attack failed for input data?
            # if adversarial[0].distance.value == 0 or adversarial[0].distance.value == np.inf:
            if (adversarial[0] is None) or (adversarial[0].distance.value == np.inf):
                robust += 1
                continue

            ## these are already adversarial example; with out adding any noise
            if (adversarial[0] .distance.value == 0):
                already_adversarial += 1
                continue

            ## the rest are successful obtained adversarial example
            perturbation = adversarial[0].unperturbed - adversarial[0].perturbed
            perturbation_norm = np.linalg.norm(perturbation)
            perturbation_norms[eps].append(perturbation_norm)
            become_adversarial += 1
            # foolrate += 1


        foolrates[eps] = {'robust': robust,
                          'become_adversarial':become_adversarial,
                          'already_adversarial':already_adversarial}.copy()

    var = {}
    for eps, item in perturbation_norms.items():
        var[eps] = {'min': str(min(item)),
                    'max': str(max(item)),
                    'robust': str(foolrates[eps]['robust']),
                    'become_adversarial': str(foolrates[eps]['become_adversarial']),
                    'already_adversarial': str(foolrates[eps]['already_adversarial']),
                    'foolrate':str((foolrates[eps]['become_adversarial'] + foolrates[eps]['already_adversarial'])/len(train_iterator.dataset))
                    }

    with open(log_path + "{}/{}/{}/wb_original_distance.json".format(chose_dataset, chose_model, model_layer), "w") as p:
        json.dump(var, p)

def wb_gravity_attacker():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = set_logger()
    if chose_dataset == 'cifar10':
        M_G = Net_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator = cifar10_loader(logger, batch_size=64)
    if chose_dataset == 'mnist':
        M_G = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator = mnist_loaders(logger, batch_size=64)

    M_G = M_G.to(device)
    M_G.eval()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    ## selecting the best weights that lead to the maximum min_link
    dff = pd.read_csv(log_path + "{}/{}/{}/distances.csv".format(chose_dataset, chose_model, model_layer))
    alfa_index = dff.iloc[dff['min'].idxmax()]['alfa']
    M_G.load_state_dict(torch.load(log_path + "{}/{}/{}/gravity_best_{}.pth".
                                   format(chose_dataset, chose_model, model_layer, alfa_index)))
    # test_loss, test_acc = evaluate(M_G, test_iterator, criterion, device)
    # print(f'\t M_G Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')

    min_total, max_total = min_max(train_iterator)
    acc_eps = {0.05:{}, 0.1:{}, 0.2:{}, 0.3:{}}

    if chose_dataset == 'cifar10':
        train_iterator, test_iterator = cifar10_loader(logger, batch_size=1)
    if chose_dataset == 'mnist':
        train_iterator, test_iterator = mnist_loaders(logger, batch_size=1)

    FM_G = foolbox.models.PyTorchModel(M_G, bounds=(min_total, max_total), num_classes=10)
    attack_M_G = foolbox.attacks.FGSM(FM_G)

    perturbation_norms = defaultdict(list)
    foolrates = defaultdict(list)
    for eps, _ in acc_eps.items():
        # teacher_adversarial_images = []
        # student_adversarial_images = []
        foolrate = 0
        robust = 0
        become_adversarial = 0
        already_adversarial = 0
        ###attack on the teacher model
        for i, (data_batch, labels_batch) in enumerate(train_iterator):
            img_numpy = data_batch.cpu().numpy()
            label_numpy = labels_batch.cpu().numpy()
            adversarial = attack_M_G(img_numpy, label_numpy,
                                     unpack=False, epsilons=[eps])

            ##Does this mean attack failed for input data?
            # if adversarial[0].distance.value == 0 or adversarial[0].distance.value == np.inf:
            if (adversarial[0] is None) or (adversarial[0].distance.value == np.inf):
                robust += 1
                continue

            if (adversarial[0].distance.value == 0):
                already_adversarial += 1
                continue

            perturbation = adversarial[0].unperturbed - adversarial[0].perturbed
            perturbation_norm = np.linalg.norm(perturbation)
            perturbation_norms[eps].append(perturbation_norm)
            become_adversarial += 1
            # foolrate += 1

        foolrates[eps] = {'robust': robust,
                          'become_adversarial':become_adversarial,
                          'already_adversarial':already_adversarial}.copy()

    var = {}
    for eps, item in perturbation_norms.items():
        var[eps] = {'min': str(min(item)),
                    'max': str(max(item)),
                    'robust': str(foolrates[eps]['robust']),
                    'become_adversarial': str(foolrates[eps]['become_adversarial']),
                    'already_adversarial': str(foolrates[eps]['already_adversarial']),
                    'foolrate':str((foolrates[eps]['become_adversarial'] + foolrates[eps]['already_adversarial'])/len(train_iterator.dataset))
                    }

    with open(log_path + "{}/{}/{}/wb_gravity_distance.json".format(chose_dataset, chose_model, model_layer), "w") as p:
        json.dump(var, p)

def bb_attacker():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = set_logger()
    if chose_dataset == 'cifar10':
        M_O = Net_fb(OUTPUT_DIM, INPUT_DIM)
        M_G = Net_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator = cifar10_loader(logger, batch_size=1)
    if chose_dataset == 'mnist':
        M_O = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        M_G = LeNet_fb(OUTPUT_DIM, INPUT_DIM)
        train_iterator, test_iterator = mnist_loaders(logger, batch_size=1)

    M_O = M_O.to(device)
    M_G = M_G.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    M_O.load_state_dict(torch.load(log_path + "{}/{}/{}/original_best.pth".
                                   format(chose_dataset, chose_model, model_layer)))

    ## selecting the best weights that lead to the maximum min_link
    dff = pd.read_csv(log_path + "{}/{}/{}/distances.csv".format(chose_dataset, chose_model, model_layer))
    alfa_index = dff.iloc[dff['min'].idxmax()]['alfa']
    M_G.load_state_dict(torch.load(log_path + "{}/{}/{}/gravity_best_{}.pth".
                                   format(chose_dataset, chose_model, model_layer, alfa_index)))

    M_O.eval()
    M_G.eval()

    # test_loss, test_acc = evaluate(M_O, test_iterator, criterion, device)
    # print(f'\t M_O Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')

    min_total, max_total = min_max(train_iterator)
    acc_eps = {0.05:{}, 0.1:{}, 0.2:{}, 0.3:{}}

    # if chose_dataset == 'cifar10':
    #     train_iterator, test_iterator = cifar10_loader(logger, batch_size=1)
    # if chose_dataset == 'mnist':
    #     train_iterator, test_iterator = mnist_loaders(logger, batch_size=1)

    FM_O = foolbox.models.PyTorchModel(M_O, bounds=(min_total, max_total), num_classes=10)
    attack_M_O = foolbox.attacks.FGSM(FM_O)


    teacher_adversarial_images = []
    student_adversarial_images = []
    for eps, _ in acc_eps.items():
        ###attack on the teacher model
        for i, (data_batch, labels_batch) in enumerate(train_iterator):
            img_numpy = data_batch.cpu().numpy()
            label_numpy = labels_batch.cpu().numpy()
            adversarial = attack_M_O(img_numpy, label_numpy,
                                     unpack=False, epsilons=[eps])

            ##Does this mean attack failed for input data?
            # if adversarial[0].distance.value == 0 or adversarial[0].distance.value == np.inf:
            if (adversarial[0] is None) or (adversarial[0].distance.value == np.inf):
                ### for the time being
                perturbation = np.zeros_like(img_numpy)
                teacher_adversarial_images.append({'perturbed': img_numpy[0],
                                                   'true_lbl': label_numpy[0],
                                                   'adv_lbl': label_numpy[0],
                                                   'perturbation': perturbation,
                                                   'attack_param': eps}.copy())
                continue

            perturbation = adversarial[0].unperturbed - adversarial[0].perturbed
            result = adversarial[0].perturbed
            teacher_adversarial_images.append({'perturbed': result,
                                               'true_lbl': label_numpy[0],
                                               'adv_lbl': adversarial[0].adversarial_class,
                                               'perturbation': perturbation,
                                                'attack_param': eps}.copy())

    ###attack on the student model
    for item in teacher_adversarial_images:
        # adversarial = attack_student(item['img'], item['true_lbl'], epsilons=2, max_epsilon=eps)
        adversarial = item['perturbed']
        # output_batch = M_G(torch.from_numpy(adversarial).to(device))
        # print(adversarial.shape)
        output_batch = M_G(torch.from_numpy(adversarial).unsqueeze(0).to(device))
        outputs = np.argmax(output_batch.detach().cpu().numpy(), axis=1)
        student_adversarial_images.append({'perturbed': adversarial,
                                           'true_lbl': item['true_lbl'],
                                           'adv_lbl': outputs[0],
                                           'perturbation': item['perturbation'],
                                           'attack_param': item['attack_param']}.copy())

    M_O_df = pd.DataFrame(teacher_adversarial_images)
    M_O_df.drop(columns=['perturbed', 'perturbation'], inplace=True)
    M_O_df.to_csv(log_path + "{}/{}/{}/M_O_df.csv".
                                   format(chose_dataset, chose_model, model_layer))

    M_G_df = pd.DataFrame(student_adversarial_images)
    M_G_df.drop(columns=['perturbed', 'perturbation'], inplace=True)
    M_G_df.to_csv(log_path + "{}/{}/{}/M_G_df.csv".
                                   format(chose_dataset, chose_model, model_layer))

main_trainer()
wb_original_attacker()
wb_gravity_attacker()
bb_attacker()
