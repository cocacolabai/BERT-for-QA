#!/usr/bin/env bash

#model saved
early=https://www.dropbox.com/s/h31vduepnnkseyn/model.zip?dl=1  #checkpoint
run=https://www.dropbox.com/s/if0yemx6wfbd8b4/bert_model.zip?dl=1
wget "${run}" -O bert_model.zip #第x個checkpoint
unzip bert_model.zip
