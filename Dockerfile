# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 debian:12

ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && \
    apt-get install -y \
        wget \
        unzip \
	software-properties-common \
        python3-full \
        python3-pip \
        jq \
        moreutils \
	git \
        tzdata && \
    rm -rf /var/lib/apt/lists/*


# Install DotNet SDK 8.0 per Microsoft's instructions
WORKDIR /opt
RUN wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y dotnet-sdk-9.0

# Download and install Dafny
WORKDIR /opt
RUN wget https://github.com/dafny-lang/dafny/releases/download/v4.9.0/dafny-4.9.0-x64-ubuntu-20.04.zip && \
    unzip dafny-4.9.0-x64-ubuntu-20.04.zip && \
    rm dafny-4.9.0-x64-ubuntu-20.04.zip

ENV PATH="/opt/dafny:/opt/dafny/dafny:$PATH"

# Download and extract Z3
RUN wget https://github.com/Z3Prover/z3/releases/download/z3-4.13.4/z3-4.13.4-x64-glibc-2.35.zip && \
    unzip z3-4.13.4-x64-glibc-2.35.zip && \
    rm z3-4.13.4-x64-glibc-2.35.zip

ENV PATH="/opt/z3-4.13.4-x64-glibc-2.35/bin:$PATH"

# Final working directory
WORKDIR /workspace

# install tensorflow
RUN python3 -m venv cav2025-artifact && \
    ./cav2025-artifact/bin/pip install tensorflow==2.13.0

RUN ./cav2025-artifact/bin/pip install tensorflow_datasets matplotlib scikit-learn

# install vulnerable gloro
RUN git clone https://github.com/klasleino/gloro && \
    cd gloro && \
    git checkout a218dcdaaa41951411b0d520581e96e7401967d7 && \
    ../cav2025-artifact/bin/python setup.py build && \
    ../cav2025-artifact/bin/python setup.py install

# copy over artifact files
RUN mkdir robustness-verifier
COPY . robustness-verifier
COPY cav2025-models.zip .
RUN unzip cav2025-models.zip