#!/bin/bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_$USER
nohup deepspeed run_translation.py \
--deepspeed /mounts/work/lavine/tools/deepspeed_transformers/transformers/tests/deepspeed/ds_config_zero2.json \
--model_name_or_path facebook/m2m100_418M \
--data_path /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/data/EU_unseen \
--per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
--output_dir /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/m2m_adapters/EU_unseen \
--overwrite_output_dir \
--num_train_epochs 20 \
--save_steps 30000 \
--do_train \
--train_adapter \
--adapter_config pfeiffer \
--max_source_length 128 \
> /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/m2m_adapters/EU_unseen/log.txt &