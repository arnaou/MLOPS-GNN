#!/bin/bash

docker build --tag=europe-west1-docker.pkg.dev/dtu-mlops-338110/gnn-mol/serve-gnn .
docker run -d --rm -it -p 8080:8080 -p 8081:8081 -m http.server --bind 0.0.0.0 --name=local-gnn europe-west1-docker.pkg.dev/dtu-mlops-338110/gnn-mol/serve-gnn 