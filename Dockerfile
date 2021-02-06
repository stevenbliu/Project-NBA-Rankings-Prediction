
ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2020.2-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

#change root to install packages
USER root

RUN pip install networkx 

RUN pip install torch

