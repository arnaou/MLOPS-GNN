#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    echo "RUNNING TORCHSERVE"
    torchserve --start --ts-config /home/model-server/config.properties --model-store=model-store --models=mol_gnn.mar
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null

#CMD ["torchserve", \
#     "--start", \
#     "--ncs"\
#     "--ts-config=/home/model-server/config.properties", \
#     "--model-store=home/model-server/model-store" \
#     "--models=mol_gnn.mar" ]