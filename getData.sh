#!/bin/bash
mkdir -p dataset/camera dataset/log

wget --continue https://archive.org/download/comma-dataset/comma-dataset.zip
mkdir -p dataset
cd dataset

unzip ../comma-dataset.zip camera/2016-01-30--11-24-51.h5 log/2016-01-30--11-24-51.h5