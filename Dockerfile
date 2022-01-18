FROM continuumio/miniconda3

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean \ 
wget \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_test.txt requirements_test.txt
COPY setup.py setup.py
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY data/ data/
COPY tests/ tests/
COPY entrypoint.sh entrypoint.sh

RUN conda init
RUN conda install pytorch=1.10.1 cpuonly -c pytorch
RUN conda install pyg=2.0.3 -c pyg -c conda-forge --yes
RUN conda install -c conda-forge rdkit=2020.09.1.0 --yes

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
WORKDIR /
#ENTRYPOINT ["python", "-u","src/data/make_dataset.py"]
ENTRYPOINT ["sh", "entrypoint.sh"]