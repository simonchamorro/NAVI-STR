


CUDA_VISIBLE_DEVICES=1 python3 train.py \
  --train_data data/train \
  --valid_data data/valid \
  --batch_ratio 1.0 \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn \
  --continue_model pth/TPS-ResNet-BiLSTM-Attn.pth \
  --valInterval 10 \
  --patience 5 \
  --select_data data \
  --no_comet \
  --apply_film \
  --skopt_n_calls 50 \
  --skopt \
  --skopt_random_state 1 \
  --skopt_cfg 'config/skopt_config.yml' \
  --experiment_name "VS-FT-CBN"

  CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --train_data data/train \
    --valid_data data/valid \
    --batch_ratio 1.0 \
    --Transformation TPS \
    --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM \
    --Prediction Attn \
    --continue_model pth/TPS-ResNet-BiLSTM-Attn.pth \
    --valInterval 10 \
    --patience 5 \
    --select_data data \
    --no_comet \
    --skopt_n_calls 50 \
    --skopt \
    --skopt_random_state 1 \
    --skopt_cfg 'config/skopt_config.yml' \
    --experiment_name "VS-FT"
