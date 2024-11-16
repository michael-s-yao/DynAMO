#!/usr/bin/bash

envname=${1:-dogambo}

# Create the conda environment.
conda create -n $envname python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate $envname

# Install robel separately.
git clone https://github.com/google-research/robel.git
cd robel
rm requirements.txt
touch requirements.txt
python -m pip install .
cd ..
rm -rf robel

# Install the package.
python -m pip install .

# Download the Design-Bench data.
cmd=$PWD
sitedir=$(python -c 'import site; exit(site.getsitepackages()[0])' 2>&1)
cd $sitedir
gdown 1SmRshNTSuMI3DxfGjfJA3463V6Hikw6_
unzip design_bench_data.zip -d design_bench_data
rm design_bench_data.zip
cd $PWD

# Train the surrogate and VAE models.
python surrogate.py -t TFBind8-Exact-v0
python surrogate.py -t GFP-Transformer-v0
python surrogate.py -t UTR-ResNet-v0
python surrogate.py -t ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0
