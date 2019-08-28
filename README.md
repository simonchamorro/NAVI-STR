# NAVI-STR
Scene Text Recognition pipeline for NAVI (Navigational Assistant for the Visually Impaired)

## Requirements

In order to install requirements, follow:

```bash
git clone https://github.com/simonchamorro/NAVI-STR
cd NAVI-STR
pip install -e .
```

## Pipeline
1. Scene text detection using yolo implementation from https://github.com/eriklindernoren/PyTorch-YOLOv3
2. Scene text recognition using this implementation https://github.com/clovaai/deep-text-recognition-benchmark

## Training
##### YOLO
    $ cd PyTorch-YOLOv3/
    $ python3 train.py --model_def config/yolov3-navi.cfg --data_config config/sevn.data --pretrained_weights weights/darknet53.conv.74

##### Starnet OCR	
    $ cd ocr/
    $ CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data data/train --valid_data data/valid  --batch_ratio 1.0 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --continue_model SAVED_MODEL --valInterval 1 --select_data data

## Testing
##### YOLO
    $ cd PyTorch-YOLOv3/

To test performance:

    $ python3 test.py --model_def config/yolov3-navi.cfg --data_config config/sevn.data --weights_path checkpoints/yolov3_ckpt_99.pth
 
To get bounding boxes and crops:

    $ python detect.py --image_folder data/sevn/test --weights_path checkpoints/yolov3_ckpt_99.pth --model_def config/yolov3-navi.cfg --class_path data/sevn/classes.names --save True

##### OCR
    $ cd ocr/

To test performance:

    $ CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_data data/dataset --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --saved_model pths/TPS-ResNet-BiLSTM-Attn.pth

To get text predictions:

    $ CUDA_VISIBLE_DEVICES=0 python3 demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder data/images/ --saved_model pths/TPS-ResNet-BiLSTM-Attn.pth

