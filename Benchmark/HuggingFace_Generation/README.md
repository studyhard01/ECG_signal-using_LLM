prepare_hf.py: convert dataset train/val files to json format for training HuggingFace models

metrics.py: calculating accuracy of the output predictions.

run_hf.py: the script for training HuggingFace models with the converted PISA dataset. This script is based on the official HuggingFace examples: https://github.com/huggingface/transformers/blob/main/examples/legacy/seq2seq/finetune_trainer.py

run_inference.py: run evaluation on the test set of PTB_XL and return accuracy.

An example script of training/evaluation is given in example_script.sh

```bash
python3 Benchmark/HuggingFace_Generation/run_hf.py \
    --model_name_or_path t5-base \
    --do_train \
    --seed=88 \
    --save_total_limit=1 \
    --train_file Dataset/base5000_30_meta_multi/train.json \
    --validation_file Dataset/base5000_30_meta_multi/val.json \
    --output_dir Pretrained_t5_ecg30meta \
    --per_device_train_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate


python3 Benchmark/HuggingFace_Generation/run_inference.py \
      -t Dataset/base5000_30_meta_multi \
      -m Benchmark/HuggingFace_Generation/model/Pretrained_t5_ecg30meta \
      -s Benchmark/HuggingFace_Generation/model/Pretrained_t5_ecg30meta_predict \
      -d ecg \
      --model_name T5 \
      -b 16
```

