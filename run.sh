#!/bin/bash
#SBATCH --mail-type=end
# std oupt
#SBATCH -o log.o

#SBATCH --partition=compute

#SBATCH --job-name="aamas"

#SBATCH --time=010:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account="research-ceg-tp"



module load miniconda3
#unset CONDA_SHLVL
#source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate aamas
cd ${HOME}/Devs/aamas
echo "Current working directory: $(pwd)"

#python ./baselines/GATRNN.py
#python run_experiments.py --mode=subset
python example.py
