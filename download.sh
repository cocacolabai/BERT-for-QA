#!/usr/bin/env bash

#model saved
early=https://www.dropbox.com/s/h31vduepnnkseyn/model.zip?dl=1  #checkpoint
wget "${early}" -O model.zip #第x個checkpoint
unzip model.zip
