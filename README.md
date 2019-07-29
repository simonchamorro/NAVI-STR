# NAVI-STR
Scene Text Recognition pipeline for NAVI (Navigational Assistant for the Visually Impaired)

## Installation
##### Clone and install requirements
    $ git clone https://github.com/simonchamorro/NAVI-STR
    $ cd NAVI-STR
    $ pip install -r requirements.txt

## Pipeline
1. Scene text detection using yolo implementation from https://github.com/eriklindernoren/PyTorch-YOLOv3
2. Scene text recognition using this implementation https://github.com/clovaai/deep-text-recognition-benchmark

## Training
##### YOLO
    $ cd PyTorch-YOLOv3/
    $ python3 train.py --model_def config/yolov3-navi.cfg --data_config config/sevn.data

## Testing
##### YOLO
    $ cd PyTorch-YOLOv3/
    $ python detect.py --image_folder data/sevn/images --weights_path checkpoints/yolov3_ckpt_99.pth --model_def config/yolov3-navi.cfg --class_path data/sevn/classes.names

##### OCR
    $ CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_data data/dataset --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --saved_model pths/TPS-ResNet-BiLSTM-Attn.pth
