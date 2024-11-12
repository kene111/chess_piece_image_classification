FROM ubuntu:22.04

EXPOSE 5000
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update -q && apt install -y git nano build-essential software-properties-common  ffmpeg libeigen3-dev libyaml-dev libfftw3-dev libtag1-dev libchromaprint-dev && \
    apt-get install -y python3 python3-pip python3-venv python3-wheel virtualenv && \
    apt-get install -y libavcodec-dev libavformat-dev libavutil-dev python3-dev python3-numpy-dev python3-numpy libsamplerate0-dev && \
    apt-get install -y screen curl wget git cmake sudo gcc g++ clang && \
    apt clean && rm -f /var/lib/apt/lists/_


COPY . ./
RUN sed -i 's/\r$//g' /start_server

RUN pip install -r /requirements.txt

workdir /app