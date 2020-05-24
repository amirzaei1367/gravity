chose_dataset = 'mnist'
chose_model ='lenet'
# chose_dataset = 'cifar10'
# chose_model ='resnet'

##cifar10 layers [-1 ...-5]
# model_layer = '-1_test'
model_layer = -1


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
EPOCHS = 200
CENT_SWITCH = 5
entropy_threshold_at = 0.85




