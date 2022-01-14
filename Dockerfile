# Base image
FROM python:3.8-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*


#Copying stuff
#install conda
FROM continuumio/miniconda3:4.9.2
RUN conda init

#install modules
RUN conda install pytorch=1.10.1 cpuonly -c pytorch
RUN conda install pyg=2.0.3 -c pyg -c conda-forge --yes
RUN conda install -c conda-forge rdkit=2020.09.1.0 --yes

COPY requirements.txt requirements.txt
COPY requirements_test.txt requirements_test.txt
COPY setup.py setup.py
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY data/ data/
COPY /tests/ tests/

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_test.txt --no-cache-dir
#Entrypoint
WORKDIR /
ENTRYPOINT ["python", "-u", "src/models/make_dataset.py"]
