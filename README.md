# NAVI-STR
Scene Text Recognition pipeline for NAVI (Navigational Assistant for the Visually Impaired)

## Installation
##### Clone and install requirements
    $ git clone https://github.com/simonchamorro/NAVI-STR
    $ cd NAVI-STR
    $ pip install -r requirements.txt

## Pipeline
1. Scene text detection using yolo implementation from https://github.com/eriklindernoren/PyTorch-YOLOv3
2. Scene text recognition

## Training
##### YOLO
    $ cd PyTorch-YOLOv3/
    $ python3 train.py --model_def config/yolov3-navi.cfg --data_config config/sevn.data

