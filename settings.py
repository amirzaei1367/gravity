chose_dataset = 'mnist'
chose_model ='lenet'
# chose_dataset = 'svhn'
# chose_dataset = 'cifar10'
# chose_model ='resnet'

##cifar10 layers [-1 ...-5]
# model_layer = '-1_momentum'
model_layer = -2


log_path = "/scratch/amirzaei/projects/gravity/log_dir/"
dataset_path = '/projects/asasan/ali/DataSets/mnist/'
# dataset_path = '/projects/asasan/ali/DataSets/cifar10/'
# dataset_path = '/projects/asasan/ali/DataSets/svhn/'
# dataset_path = '/projects/asasan/ali/DataSets/fmnist/'
# dataset_path = '/projects/asasan/ali/DataSets/cifar10/'

cifar_batch = 128
mnist_batch = 64
svhn_batch = 128
fmnist_batch = 64
# BATCH_SIZE = 1
SEED = 1234
OUTPUT_DIM = 10
INPUT_DIM = 1
EPOCHS = 250
CENT_SWITCH = 10
entropy_threshold_at = 0.85
adv_trainer = None

# DBSCN HYPER Parameteres:
DBSCAN_EPS_i = 0.9
DBSCAN_SAMPLES_i = 30
Circle_Diam_i = 100

DBSCAN_EPS_1 = 0.3
DBSCAN_SAMPLES_1 = 20
Circle_Diam_1 = 50


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet110': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110-1d1ed7c2.th',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


