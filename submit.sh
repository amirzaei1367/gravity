#!/bin/bash

#SBATCH --job-name gr_m_2
#SBATCH --partition gpuq
#SBATCH --nodes 1
###SBATCH --nodelist=NODE076
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
###SBATCH --array=98-99
####SBATCH --output=/scratch/amirzaei/cluster_kd/kd_2clusters_%a/cluster0/extractor_%N_%j.log 
###SBATCH --output=/scratch/amirzaei/cluster_kd/kd_2clusters_%a/cluster0/extractor.log 
#SBATCH --output=/scratch/amirzaei/projects/gravity/log_dir/%x.log
###SBATCH --output=/scratch/amirzaei/cifar100/extractor_%N_%j.log 
###SBATCH --output=/scratch/amirzaei/cluster_kd/kd_2clusters_%a/cluster1/extractor_%N_%j.log 
#SBATCH --mem 50000
##SBATCH --qos=hhqos
###SBATCH --reservation=amirzaei_75
###SBATCH --time=0-12:0

##module load cuda80/toolkit/8.0.44
##module purge
#export GIT_PYTHON_REFRESH=quiet
#module load pytorch/1.4.0-p36
module load python/3.6.7
source /home/amirzaei/venv/bin/activate
##
##module load python/python2.7
##module load pytorch/0.4.1-py36




###python /scratch/amirzaei/cifar100/extractor.py

##python /scratch/amirzaei/cluster_kd/generated_train/${SLURM_ARRAY_TASK_ID}.py  --model_dir  /scratch/amirzaei/cluster_kd/kd_2clusters_${SLURM_ARRAY_TASK_ID}/cluster0/ 
                                                                                    #--restore_file /scratch/amirzaei/baseline_resnet/base_resnet18/best
																					
python /home/amirzaei/projects/gravity/main.py																					
##python /scratch/amirzaei/cluster_kd/generated_train/${SLURM_ARRAY_TASK_ID}.py  --model_dir  /scratch/amirzaei/cluster_kd/kd_2clusters_${SLURM_ARRAY_TASK_ID}/cluster1/ 
# python /home/amirzaei/baseline_resnet/extractCifar10.py																					
                                                                                    
##python /home/amirzaei/distillation/knowledge-distillation-pytorch-master/train.py  --model_dir  /scratch/amirzaei/dist_restore_file_2/base_resnet18/                                                                                                                         ##--restore_file /scratch/amirzaei/dist_restore_file
                                                                                    
##source /home/amirzaei/distillation/knowledge-distillation-pytorch-master/distillation_env/bin/deactivate

# # scipy==1.0.0
# # numpy==1.14.0
# # Pillow==5.0.0
# # tabulate==0.8.2
# # tensorflow==1.7.0rc0
# # torch==0.3.0.post4
# # torchvision==0.2.0
# # tqdm==4.19.8
# # torchnet
