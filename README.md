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
* Code version control
* Data Version control
* Good practices for organizing an ML project
* Insure reproducibility
* Logging in order to perform virtual experimentation such hyper-parameter optimization
* (Hopefully) Distributed training
* (Definitely) Deployment of the model so the public can use it





















<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
