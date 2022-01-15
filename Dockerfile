FROM python:3.8-slim
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

FROM continuumio/miniconda3
#install modules
RUN conda init
RUN conda install pytorch=1.10.1 cpuonly -c pytorch
RUN conda install pyg=2.0.3 -c pyg -c conda-forge --yes
RUN conda install -c conda-forge rdkit=2020.09.1.0 --yes

FROM python:3.8-slim

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
CMD ["python", "-u", "src/models/make_dataset.py"]
