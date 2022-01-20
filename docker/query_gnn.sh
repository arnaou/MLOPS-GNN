#!/bin/sh
echo "Welcome to our AI!!!"
echo "Querying Cloud Model"
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json; charset=utf-8" -d @docker/instances.json https://europe-west1-ml.googleapis.com/v1/projects/dtu-mlops-338110/models/gnn_mol/versions/v4:predict
echo "\nThanks for playing"