from setuptools import find_packages, setup

setup(
    name='GNN-Mol',
    long_description='The model of interest is a Graph Neural Network developed in the Pytorch-Geometric framework. The choice of model is tentative but various implementations already exists. The model is trained on two independent dataset to predict molecular properties such as the Aqueous solubility (ESOL) and the melting point of organic compounds. As input to the models a graph with molecular attributes will be generated using the SMILEs notation and an open-source cheminformatics package RDKit.',
    packages=find_packages(),
    version='0.1.0',
    description='repository for ML-OPS project',
    author='Adem R. N. Aouichaoui, Aswin Anil Varkey, Simon Goldhofer',
    license='BSD-3',
    author_email="arnaou@kt.dtu.dk",
    install_requires=[
        "Click",
        "python-dotenv",
        "rdkit",
        "setuptools",
        "torch",
        "torch_geometric",
    ],
    python_requires='>=3.8',
)
