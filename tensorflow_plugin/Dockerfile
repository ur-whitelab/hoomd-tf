FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER Andrew White <andrew.white@rochester.edu>

RUN apt-get update && apt-get install -y git cmake gcc g++ sqlite3 wget libsqlite3-dev && apt-get clean

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /srv/miniconda.sh
RUN chmod +x /srv/miniconda.sh && /srv/miniconda.sh -b -p /usr/local/miniconda
ENV PATH="/usr/local/miniconda/bin:$PATH"

RUN conda config --add channels glotzer && conda install -y fresnel gsd freud &&  conda clean -a
RUN conda install -y Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        networkx \
        scipy \
        pygraphviz \
        tqdm \
        && \
python -m ipykernel.kernelspec && \
conda clean -a

#use pip, since conda distro is unofficial
RUN pip install tensorflow-gpu

RUN git clone --recursive https://bitbucket.org/glotzer/hoomd-blue /srv/hoomd-blue
WORKDIR /srv/hoomd-blue

#Add plugin
RUN mkdir -p hoomd/tensorflow_plugin
ADD . hoomd/tensorflow_plugin

ENV PYTHONPATH=$PYTHONPATH:/srv/hoomd-blue/build
ENV PATH="/usr/local/bin:$PATH"

EXPOSE 8888

WORKDIR "/notebooks"

CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0"]

#can't run this because we need gpu
#RUN export SOFTWARE_ROOT=/usr/local && mkdir build && cd build && \
#        cmake ../ -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python \
#         -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=OFF \
#         && make -j 6

#cmake .. -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=OFF -DBUILD_HPMC=off -DBUILD_CGCMM=off -DBUILD_MD=off -DBUILD_METAL=off -DBUILD_TESTING=off -DBUILD_DEPRECATED=off