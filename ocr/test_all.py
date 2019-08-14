import os
import random
import subprocess

seeds = [0, 1, 2, 3, 4]

# VS
# Testing
for seed in seeds:
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1 python3 test.py --eval_data data/test \
        --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM  \
        --Prediction Attn --saved_model pth/TPS-ResNet-BiLSTM-Attn.pth             \
        --experiment_name VS-seed-{seed} --seed {seed}", shell=True)

# VS + matching
# Testing
for seed in seeds:
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1 python3 test.py --eval_data data/test \
        --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM  \
        --Prediction Attn --saved_model pth/TPS-ResNet-BiLSTM-Attn.pth             \
        --ed_condition --experiment_name VS-matching-seed-{seed} --seed {seed}", shell=True)

# VS + fine-tuning
# Training
for seed in seeds:
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data data/train \
        --valid_data data/valid  --batch_ratio 1.0 --Transformation TPS               \
        --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn        \
        --select_data data --continue_model pth/TPS-ResNet-BiLSTM-Attn.pth            \
        --no_comet --experiment_name VS-ft-seed-{seed} --manualSeed {seed}", shell=True)

# Testing
for seed in seeds:
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1 python3 test.py --eval_data data/test       \
        --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM        \
        --Prediction Attn --saved_model saved_models/VS-ft-seed-{seed}/best_accuracy.pth \
        --experiment_name VS-ft-seed-{seed} --seed {seed}", shell=True)

# VS + fine-tuning + matching
# Testing
for seed in seeds:
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1 python3 test.py --eval_data data/test       \
        --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM        \
        --Prediction Attn --saved_model saved_models/VS-ft-seed-{seed}/best_accuracy.pth \
        --ed_condition --experiment_name VS-ft-matching-seed-{seed} --seed {seed}", shell=True)

# VS + CBN
# Training
for seed in seeds:
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data data/train \
        --valid_data data/valid  --batch_ratio 1.0 --Transformation TPS               \
        --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn        \
        --select_data data --continue_model pth/TPS-ResNet-BiLSTM-Attn.pth --no_comet \
        --apply_film --experiment_name VS-CBN-seed-{seed} --manualSeed {seed}", shell=True)

# Testing
for seed in seeds:
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1 python3 test.py --eval_data data/test        \
        --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM         \
        --Prediction Attn --saved_model saved_models/VS-CBN-seed-{seed}/best_accuracy.pth \
        --apply_film --experiment_name VS-CBN-seed-{seed} --seed {seed}", shell=True)

# VS + CBN + matching
# Testing
for seed in seeds:
    subprocess.call(f"CUDA_VISIBLE_DEVICES=1 python3 test.py --eval_data data/test        \
        --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM         \
        --Prediction Attn --saved_model saved_models/VS-CBN-seed-{seed}/best_accuracy.pth \
        --apply_film --ed_condition --experiment_name VS-CBN-matching-seed-{seed} --seed {seed}", shell=True)


