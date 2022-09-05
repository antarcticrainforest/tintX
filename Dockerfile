FROM condaforge/mambaforge:latest

ARG NB_USER="tintx"
ARG NB_UID="1000"

ENV USER=${NB_USER} \
    HOME=/tmp/${NB_USER} \
    NB_GID=${NB_UID} \
    NB_GROUP=${NB_USER} \
    DATA_FILES=/tmp/$NB_USER/.data

COPY . /tmp/tint_clone

RUN set -e && \
  groupadd -r --gid "$NB_GID" "$NB_GROUP" && \
  adduser --uid "$NB_UID" --gid "$NB_GID" --gecos "Default user" \
  --shell /bin/bash --disabled-password "$NB_USER" --home $HOME && \
  mamba install -c conda-forge -y tintx ipykernel jupyterlab notebook bash_kernel &&\
  cp -r /tmp/tint_clone/docs/source/_static/data $HOME/.data &&\
  for i in $(ls /tmp/tint_clone/docs/source/*.ipynb);do sed -i "s/\.html/\.ipynb/g" $i ;done &&\
  cp /tmp/tint_clone/docs/source/*.ipynb $HOME/ &&\
  cp /tmp/tint_clone/.Readme.ipynb $HOME/Readme.ipynb &&\
  rm -fr /tmp/tint_clone &&\
  chown -R $NB_USER:$NB_GROUP $HOME

USER $NB_USER
WORKDIR $HOME
RUN set -e &&\
  mamba run python3 -m bash_kernel.install &&\
  mamba run python3 -m ipykernel install --name tintx --user \
    --env DATA_FILES $DATA_FILES --display-name "tintX kernel"
