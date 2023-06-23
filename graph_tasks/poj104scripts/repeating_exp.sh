#!/bin/bash -l
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J jc
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
conda activate codebert
low=1
hgh=1000
for r in $(seq 3)
do
rand=$((low + RANDOM%(1+hgh-low)))
for layer in $(seq 12 -1 0)
do
bash job_codebert.sh ast  $layer $rand $r
bash job_codebert.sh cfg  $layer $rand $r
bash job_codebert.sh cdg  $layer $rand $r
bash job_codebert.sh ddg  $layer $rand $r
done 
done 

