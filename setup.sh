#!/usr/bin/bash

envname=${1:-dynamo}

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
gdown 1zI7HQQ4CoJYTTbKnmrS-GInbwYmDEl0F
unzip design_bench_data.zip -d design_bench_data
rm design_bench_data.zip
cd $cmd

# Download story generation oracle dependency.
git submodule update --init --recursive
gdown 1HIuE8iqYWQ2t01bTJJnxNN1QV-TcCqbx
unzip saved_latent_models.zip -d latent-diffusion-for-language/
rm saved_latent_models.zip

# Download the Guacamol dataset.
wget https://figshare.com/ndownloader/files/13612757 -O train.smiles
wget https://figshare.com/ndownloader/files/13612766 -O val.smiles

# Train the surrogate and VAE models.
python surrogate.py -t TFBind8-Exact-v0
python surrogate.py -t UTR-ResNet-v0
python surrogate.py -t ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0
python surrogate.py -t PenalizedLogP-Exact-v0
python surrogate.py -t Superconductor-RandomForest-v0
python surrogate.py -t DKittyMorphology-Exact-v0
python surrogate.py -t StoryGen-Exact-v0
