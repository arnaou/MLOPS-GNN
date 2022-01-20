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

The model selected is : *Attentive fingerprint*
* [Paper](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959) 
* [GitHub](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py)
* [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html)


# Setup Conda Environment:

Update Conda: `conda update --yes conda`

Set up conda env: `conda create -n GNN-Mol python=3.8 --yes`

Activate conda env: `conda activate GNN-MOL`

Install conda packages:

`conda install -c pytorch pytorch=1.10.1 cpuonly`

`conda install pyg=2.0.3 -c pyg -c conda-forge --yes`

`conda install -c conda-forge rdkit=2020.09.1.0 --yes`

Install pip packages:

`pip install -r requirements.txt`

`pip install -r requirements_test.txt`

# Docker

Create docker: `docker build -t gnn:latest .`

Run Docker from entrypoint: `docker run gnn`

Run Docker with shell as entrypoint: `docker run -it --entrypoint sh gnn`

Upload to Dockerhub: `docker container commit CONTAINER-ID gnn-mol-latest` (Get CONTAINER-ID with `docker ps -a`), then use docker extension to push.

To Pull: `docker pull 123456789523544/gnn-mol-latest`

# Deployment

Build from Dockerfile: `docker build -f docker/Dockerfile --tag=europe-west1-docker.pkg.dev/dtu-mlops-338110/gnn-mol/serve-gnn .`

Run locally: `docker run --rm -it -p 8080:8080 -p 8081:8081 --name=local-gnn europe-west1-docker.pkg.dev/dtu-mlops-338110/gnn-mol/serve-gnn`

Test locally: `cat > instances.json <<END
{
  "instances": [
    {
      "data": "CCCCCCCO"
    }
  ]
}
END`

`curl -X POST \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @instances.json \
  localhost:8080/predictions/gnn_mol`

Push to GCloud Artifact Registry: `gcloud auth configure-docker europe-west1-docker.pkg.dev`

`docker push europe-west1-docker.pkg.dev/dtu-mlops-338110/gnn-mol/serve-gnn`

Create a model version: `gcloud beta ai-platform versions create v2 \
  --region=europe-west1 \
  --model=gnn_mol \
  --machine-type=n1-standard-4 \
  --image=europe-west1-docker.pkg.dev/dtu-mlops-338110/gnn-mol/serve-gnn \
  --ports=8080 \
  --health-route=/ping \
  --predict-route=/predictions/gnn_mol`

Test Deployment: `curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @instances.json \
  https://europe-west1-ml.googleapis.com/v1/projects/dtu-mlops-338110/models/gnn_mol/versions/v4:predict`



More help: https://cloud.google.com/ai-platform/prediction/docs/getting-started-pytorch-container


# Usage










<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
