MLOPS-GNN
==============================

repository for ML-OPS project

Project Description
------------
The overall aim of the project is to apply various ML-OPS tools to facilitate
the further development and maintenance of a deep-learning model.

The model of interest is a Graph Neural Network developed in the Pytorch-Geometric
framework. The choice of model is tentative but various implementations already exists.
The model is trained on two independent dataset to predict molecular properties such as the
Aqueous solubility (ESOL) and the melting point of organic compounds. As input to the models a 
graph with molecular attributes will be generated using the SMILEs notation and an open-source
cheminformatics package RDKit.

List of ML-OPS tools to be incorporated throughout the execution of the project are:
* Structuring of Project using cookiecutter
* Code version control (git)
* Data Version control (dvc)
* Package management using conda virtual environments
* Ensure reproducibility using MLFlow or Comet.ml, including logging, model registry and tracking
* (Optional) Perform virtual experimentation such as hyper-parameter optimization using sweeps (Weights&Biases)
* Comply with Pep8 standards, fix code using black or yapf
* Create small unit tests for data pre-processing and training
* (Optional) Distributed training
* Deployment and monitoring of the model

The model selected is : *Hierarchical Inter-Message Passing for Learning on Molecular Graphs*
* [Paper](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959) 
* [GitHub](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py)
* [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html)


To Setup Conda Environment:

Update Conda: `conda update --yes conda`

Set up conda env: `conda create -n GNN-Mol python=3.8 --yes`

Activate conda env: `conda activate GNN-MOL`

Install conda packages:

`conda install -c pytorch pytorch=1.10.1`

`conda install pyg=2.0.3 -c pyg -c conda-forge --yes`

`conda install -c conda-forge rdkit=2020.09.1.0 --yes`

Install pip packages:

`pip install -r requirements.txt`

`pip install -r requirements_test.txt`















<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
