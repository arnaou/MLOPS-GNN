FROM continuumio/miniconda3 AS conda_setup

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean \ 
wget \
&& rm -rf /var/lib/apt/lists/*
RUN apt-get update \
    && apt-get install -y software-properties-common
RUN apt install --no-install-recommends -y openjdk-11-jre-headless

RUN mkdir -p /root/gnn-mol
COPY requirements.txt requirements.txt
COPY requirements_test.txt requirements_test.txt
COPY setup.py setup.py


RUN conda init
RUN conda install pytorch=1.10.1 cpuonly -c pytorch
RUN conda install pyg=2.0.3 -c pyg -c conda-forge --yes
RUN conda install -c conda-forge rdkit=2020.09.1.0 --yes
RUN conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch

RUN pip install --no-cache-dir captum python-dotenv[cli] torchserve torch-model-archiver torch-workflow-archiver
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_test.txt --no-cache-dir


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

WORKDIR /root/gnn-mol
RUN dvc pull
WORKDIR /


#COPY entrypoint.sh root/gnn-mol/entrypoint.sh
COPY src/models/model.py model_handler.py models/checkpoint.pth /home/model-server/
COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:8080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:8081" >> /home/model-server/config.properties
RUN printf "\nmetrics_address=http://0.0.0.0:8082" >> /home/model-server/config.properties
RUN printf "\nnumber_of_netty_threads=32" >> /home/model-server/config.properties
RUN printf "\njob_queue_size=1000" >> /home/model-server/config.properties
RUN printf "\nmodel_store=/home/model-server/model-store" >> /home/model-server/config.properties

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp \
    && chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server


RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

RUN torch-model-archiver \
  --model-name=mol_gnn \
  --version=1.0 \
  --model-file=/home/model-server/model.py \
  --serialized-file=/home/model-server/checkpoint.pth \
  --handler=/home/model-server/model_handler.py \
  --export-path=/home/model-server/model-store


EXPOSE 8080 8081 8082 7070 7071
USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]


#CMD ["torchserve", \
#     "--start", \
#     "--ncs"\
#     "--ts-config=/home/model-server/config.properties", \
#     "--model-store=home/model-server/model-store" \
#     "--models=mol_gnn.mar" ]



