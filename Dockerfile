FROM continuumio/miniconda3

ARG SECRET_KEY
RUN echo SECRET_KEY
ENV SECRET_KEY $SECRET_KEY
RUN echo "$SECRET_KEY\n$SECRET_KEY"
RUN echo "$SECRET_KEY\n$SECRET_KEY" > file.txt
RUN echo SECRET_KEY > file1.txt
RUN echo "$SECRET_KEY\n$SECRET_KEY1" > file2.txt
RUN echo "$SECRET_KEY\n$SECRET_KEY2" > file3.txt

#RUN apt update && \
#apt install --no-install-recommends -y build-essential gcc && \
#apt clean \ 
#wget \
#&& rm -rf /var/lib/apt/lists/*
RUN mkdir /root/gnn-mol
COPY requirements.txt requirements.txt
COPY requirements_test.txt requirements_test.txt
COPY setup.py setup.py
# RUN conda init
# RUN conda install pytorch=1.10.1 cpuonly -c pytorch
# RUN conda install pyg=2.0.3 -c pyg -c conda-forge --yes
# RUN conda install -c conda-forge rdkit=2020.09.1.0 --yes
# RUN pip install -r requirements.txt --no-cache-dir
# RUN pip install -r requirements_test.txt --no-cache-dir
# RUN pip install python-dotenv[cli]
# RUN wget -nv \
#     https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
#     mkdir /root/tools && \
#     tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
#     rm google-cloud-sdk.tar.gz && \
#     /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
#         --path-update=false --bash-completion=false \
#         --disable-installation-options && \
#     rm -rf /root/.config/* && \
#     ln -s /root/.config /config && \
#     # Remove the backup directory that gcloud creates
#     rm -rf /root/tools/google-cloud-sdk/.install/.backup
# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin -
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg
ENV PYTHONPATH "${PYTHONPATH}:/root/project"
#Entrypoint
COPY src/ root/gnn-mol/src/
COPY models/ root/gnn-mol/models/
COPY reports/ root/gnn-mol/reports/
COPY tests/ root/gnn-mol/tests/
COPY entrypoint.sh root/gnn-mol/entrypoint.sh
WORKDIR /root/gnn-mol/
#ENTRYPOINT ["python", "-u","src/data/make_dataset.py"]
ENTRYPOINT ["sh", "entrypoint.sh"]