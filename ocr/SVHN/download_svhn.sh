#!/bin/bash

export SVHN_DIR=$HOME'/NAVI-STR/ocr/SVHN'

# Download data
if [ ! -f $SVHN_DIR'/train.tar.gz' ]; then
    echo "Downloading files for the training set!"
    wget -P $SVHN_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

if [ ! -f $SVHN_DIR'/extra.tar.gz' ]; then
    echo "Downloading files for the extra set!"
    wget -P $SVHN_DIR http://ufldl.stanford.edu/housenumbers/extra.tar.gz
fi

if [ ! -f $SVHN_DIR'/test.tar.gz' ]; then
    echo "Downloading files for the test set!"
    wget -P $SVHN_DIR http://ufldl.stanford.edu/housenumbers/test.tar.gz
fi

# Unzip tar.gz
if [ ! -d $SVHN_DIR'/train' ]; then
    tar xvzf $SVHN_DIR'/train.tar.gz'
fi

if [ ! -d $SVHN_DIR'/extra' ]; then
    tar xvzf $SVHN_DIR'/extra.tar.gz'
fi

if [ ! -d $SVHN_DIR'/test' ]; then
    tar xvzf $SVHN_DIR'/test.tar.gz'
fi
