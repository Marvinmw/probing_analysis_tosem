#!/bin/bash -l
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J j1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
conda activate codebert
low=1
hgh=1000

for r in $(seq 1 4)
do
    rand=$((low + RANDOM%(1+hgh-low)))
    for layer in $(seq 12 -1 0)
    do
    if [ ! -f ./pos_res_java_graph/plm/saved_graphcodebertmodels/${r}/layer_${layer}/predictions_performance_test.json ]; then
        bash run_java_graph.sh $layer $rand $r
        echo ./pos_res_java_graph/plm/saved_graphcodebertmodels/${r}/layer_${layer}/predictions_performance_test.json
    fi

    if [ ! -f ./pos_res_poj/random/saved_graphcodebertmodels/${r}/layer_${layer}/predictions_performance_test.json ]; then
        bash run_poj_graph.sh $layer $rand $r
        echo ./pos_res_poj/random/saved_graphcodebertmodels/${r}/layer_${layer}/predictions_performance_test.json
    fi
    done
done
