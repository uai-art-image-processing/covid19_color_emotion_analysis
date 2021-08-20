#/bin/bash

rm -rf logs
tensorboard --bind_all --logdir=./logs
