FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER Andrew White <andrew.white@rochester.edu>

RUN apt-get update && apt-get install -y git cmake gcc\
     sqlite3 wget libsqlite3-dev bash-completion g++ \
     zlib1g-dev libtcmalloc-minimal4 libopenmpi-dev \
     openmpi-bin && apt-get clean

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /srv/miniconda.sh
RUN chmod +x /srv/miniconda.sh && /srv/miniconda.sh -b -p /usr/local/miniconda
ENV PATH="/usr/local/miniconda/bin:$PATH"

RUN conda create -n py36 python=3.6
ENV PATH="/usr/local/miniconda/envs/py36/bin:$PATH"

#use pip, since conda distro is unofficial
RUN pip install tensorflow-gpu
#RUN pip install tensorflow

RUN conda config --add channels glotzer && \
        conda install -y fresnel gsd freud &&  conda clean -a
RUN conda install -y Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        networkx \
        scipy \
        pygraphviz \
        tqdm &&\
        conda clean -a

RUN git clone --recursive https://bitbucket.org/glotzer/hoomd-blue /srv/hoomd-blue
WORKDIR /srv/hoomd-blue

#Add plugin
RUN mkdir -p htf
RUN cd hoomd && ln -s ../htf htf
#ADD . hoomd/htf


ENV PYTHONPATH=$PYTHONPATH:/srv/hoomd-blue/build
ENV PATH="/usr/local/bin:$PATH"

EXPOSE 8888

WORKDIR "/notebooks"

CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0"]
