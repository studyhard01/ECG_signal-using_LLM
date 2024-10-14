#!/usr/bin/env bash



python3 run_hf.py \
    --model_name_or_path google_pegasus-xsum\
    --do_train \
    --seed=88 \
    --save_total_limit=1 \
    --train_file SG/train.json \
    --validation_file SG/val.json \
    --output_dir SG_Pretrained_Pegasus \
    --per_device_train_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate

python3 /run_inference.py \
      -t SG \
      -m SG_Pretrained_Pegasus \
      -s G_Pretrained_Pegasus_pred \
      -d SG \
      --model_name pegasus \
      -b 16


python3 run_hf.py \
    --model_name_or_path facebook/bart-large\
    --do_train \
    --seed=88 \
    --save_total_limit=1 \
    --train_file /tf/hsh/new_ecg/train.json \
    --validation_file /tf/hsh/new_ecg/val.json \
    --output_dir SG_Pretrained_Pegasus1_ecg \
    --per_device_train_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate


python3 run_inference.py \
      -t /tf/hsh/new_ecg \
      -m /tf/yj/promptcast/PISA/Benchmark/HuggingFace_Generation/SG_Pretrained_bart_ecg \
      -s /tf/hsh/PISA/Benchmark/HuggingFace_Generation/SG_Pretrained_bart_ecg_predict \
      -d ecg \
      --model_name bart \
      -b 16


python3 run_hf.py \
    --model_name_or_path facebook/bart-base\
    --do_train \
    --seed=88 \
    --save_total_limit=1 \
    --train_file /tf/hsh/new_ecg/train.json \
    --validation_file /tf/hsh/new_ecg/val.json \
    --output_dir SG_Pretrained_bart_ecg \
    --per_device_train_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate


python3 run_inference.py \
      -t /tf/hsh/new_ecg \
      -m /tf/yj/promptcast/PISA/Benchmark/HuggingFace_Generation/SG_Pretrained_bart_ecg \
      -s /tf/hsh/PISA/Benchmark/HuggingFace_Generation/SG_Pretrained_bart_ecg_predict \
      -d ecg \
      --model_name bart \
      -b 16




python3 run_hf.py \
    --model_name_or_path facebook/bart-base\
    --do_train \
    --seed=88 \
    --save_total_limit=1 \
    --train_file /tf/LHI/PromptCast/new_new_ecg_prompt/train.json \
    --validation_file /tf/LHI/PromptCast/new_new_ecg_prompt/val.json \
    --output_dir SG_Pretrained_bart_ecg_1 \
    --per_device_train_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate









