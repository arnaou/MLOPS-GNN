#! /bin/sh.
echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
echo Loading environment.yml and installing packages
conda init bash
conda env create -f environment.yml
#conda activate MLOps-GNN
echo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
echo Installing rdkit
conda install -c conda-forge rdkit