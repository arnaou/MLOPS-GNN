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
* [Paper](https://arxiv.org/abs/2006.12179) 
* [GitHub](https://github.innominds.com/rusty1s/himp-gnn)





















<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
