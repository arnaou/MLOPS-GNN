#!/bin/sh
echo "Welcome to our AI!!!"
read -p "Enter smile structure: "  sm
cat > query.json <<END
{
 "instances": [
   {
     "data": "$sm"
   }
 ]
}
END
echo "Querying Cloud Model"
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json; charset=utf-8" -d @query.json https://europe-west1-ml.googleapis.com/v1/projects/dtu-mlops-338110/models/gnn_mol/versions/v2:predict
echo "\nThanks for playing"