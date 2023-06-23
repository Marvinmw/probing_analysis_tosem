#!/bin/bash -l
#SBATCH -n 5
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --mem=80GB
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J h1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
#SBATCH -C volta32
conda activate codebert
low=1
hgh=1000

for r in $(seq 4)
do
rand=$((low + RANDOM%(1+hgh-low)))
for i in $(seq 12 -1 0)
do

bash job_codebert.sh  ddg $i $rand $r
bash job_codebert.sh  cdg $i $rand $r

done 
done



