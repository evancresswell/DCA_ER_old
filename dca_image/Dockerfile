FROM debian:buster-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ENV PYTHON_VERSION 3.7

#RUN useradd -ms /bin/bash test_user

#RUN usermod -u 1001 test_user
#RUN groupmod -g 1001 test_user

RUN apt-get update --fix-missing && \
    apt-get install -y wget build-essential bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion sudo && \
    apt-get clean

#RUN echo "test_user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# conda and test dependencies
RUN /opt/conda/bin/conda install -y -c defaults \
    conda conda-package-handling \
    python=$PYTHON_VERSION pycosat requests ruamel_yaml cytoolz \
    anaconda-client nbformat make \
    pytest pytest-cov codecov radon pytest-timeout mock responses pexpect \
    flake8 && \
    /opt/conda/bin/conda clean --all --yes

RUN sudo /opt/conda/bin/pip install llvmlite --ignore-installed
RUN sudo /opt/conda/bin/pip install pydca

#USER test_user

#ENTRYPOINT [ "/bin/bash" ]
CMD [ "/bin/bash" ]




