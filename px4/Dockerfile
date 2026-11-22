# PX4 SITL + Gazebo Harmonic, headless, Ubuntu 22.04 (arm64 on Apple Silicon)
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release sudo git wget python3 python3-pip python3-venv tzdata \
      build-essential cmake ninja-build ccache g++ gcc make pkg-config unzip zip rsync file \
      mesa-utils libgl1-mesa-dri libegl1 libegl-mesa0 xvfb && rm -rf /var/lib/apt/lists/*
# Gazebo Harmonic from OSRF
RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && apt-get install -y --no-install-recommends gz-harmonic && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash px4 && echo "px4 ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/px4
USER px4
WORKDIR /home/px4
