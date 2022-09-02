FROM condaforge/mambaforge:latest

ARG NB_USER="tintx"
ARG NB_UID="1000"

ENV USER=${NB_USER} \
    HOME=/tmp/${NB_USER} \
    NB_GID=${NB_UID} \
    NB_GROUP=${NB_USER} \
    DATA_FILES=/tmp/$NB_USER/.data
USER root

RUN set -e && \
  groupadd -r --gid "$NB_GID" "$NB_GROUP" && \
  adduser --uid "$NB_UID" --gid "$NB_GID" --gecos "Default user" \
  --shell /bin/bash --disabled-password "$NB_USER" --home $HOME
COPY . /tmp/tint
WORKDIR /tmp/tint
RUN set -e && \
    mamba install -y cartopy ffmpeg &&\
    mamba run pip install .[docs,test] --no-cache-dir ipykernel jupyterlab \
    notebook bash_kernel &&\
    cp -r docs/source/_static/data $HOME/.data &&\
    cp -r docs/source/*.ipynb $HOME/ &&\
    cp .Readme.ipynb $HOME/Readme.ipynb &&\
    cd $HOME &&\
    rm -fr /tmp/tint

WORKDIR $HOME
USER $NB_USER
RUN set -e && \
    mamba run python3 -m bash_kernel.install &&\
    mamba run python3 -m ipykernel install --name plotting --user --env DATA_FILES $DATA_FILES
