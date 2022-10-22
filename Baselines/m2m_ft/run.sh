#!/bin/bash
# m2m_ft
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_$USER
nohup deepspeed run_translation.py \
--deepspeed /mounts/work/lavine/tools/deepspeed_transformers/transformers/tests/deepspeed/ds_config_zero2.json \
--model_name_or_path facebook/m2m100_418M \
--per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
--data_path /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/data/EU_unseen \
--output_dir /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/m2m_ft/EU_unseen \
--overwrite_output_dir \
--num_train_epochs 3 \
--save_steps 20000 \
--do_train \
--max_source_length 128 \
> /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/m2m_ft/EU_unseen/log.txt &

# m2m_tag
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_$USER
nohup deepspeed run_translation.py \
--deepspeed /mounts/work/lavine/tools/deepspeed_transformers/transformers/tests/deepspeed/ds_config_zero2.json \
--model_name_or_path facebook/m2m100_418M \
--per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
--data_path /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/data/EU_unseen_tags \
--output_dir /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/m2m_tags/EU_unseen \
--overwrite_output_dir \
--num_train_epochs 3 \
--save_steps 20000 \
--do_train \
--max_source_length 128 \
> /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/m2m_tags/EU_unseen/log.txt &