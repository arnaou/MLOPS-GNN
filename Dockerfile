FROM continuumio/miniconda3 AS conda_setup

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean \ 
wget \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /root/gnn-mol
COPY requirements.txt requirements.txt
COPY requirements_test.txt requirements_test.txt
COPY setup.py setup.py


RUN conda init
RUN conda install pytorch=1.10.1 cpuonly -c pytorch
RUN conda install pyg=2.0.3 -c pyg -c conda-forge --yes
RUN conda install -c conda-forge rdkit=2020.09.1.0 --yes
RUN conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch

RUN pip install torchserve torch-model-archiver torch-workflow-archiver
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_test.txt --no-cache-dir
RUN pip install python-dotenv[cli]

RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin -

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg
#Entrypoint

COPY src/ root/gnn-mol/src/
COPY models/ root/gnn-mol/models/
COPY reports/ root/gnn-mol/reports/
COPY data/ root/gnn-mol/data/
COPY tests/ root/gnn-mol/tests/
COPY .dvc/ root/gnn-mol/.dvc/
COPY .git/ root/gnn-mol/.git
COPY data.dvc root/gnn-mol/data.dvc


#FROM pytorch/torchserve:0.3.0-cpu AS serve_deploy

COPY entrypoint.sh root/gnn-mol/entrypoint.sh
COPY src/models/model.py model_handler.py models/checkpoint.pth /home/model-server/

#WORKDIR /
#COPY --from=conda_setup root/gnn-mol/ /home/gnn-mol/
#COPY --from=conda_setup /usr/bin/conda_setup /usr/bin/conda_setup 
#WORKDIR /home/model-server/

#USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
#USER model-server
RUN mkdir home/model-server/model-store

RUN torch-model-archiver \
  --model-name=mol_gnn \
  --version=1.0 \
  --model-file=/home/model-server/model.py \
  --serialized-file=/home/model-server/checkpoint.pth \
  --handler=/home/model-server/model_handler.py \
  --export-path=/home/model-server/model-store

RUN mv mol_gnn.mar home/model-server/model-store
WORKDIR /root/gnn-mol/
RUN dvc pull

CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--model-store=home/model-server/model-store" \
     "--models=mol_gnn=mol_gnn.mar"]


#WORKDIR /root/gnn-mol/

#ENTRYPOINT ["python", "-u","src/data/make_dataset.py"]
#CMD ["sh", "entrypoint.sh"]
