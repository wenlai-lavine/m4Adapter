#!/bin/bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_$USER
nohup deepspeed --num_gpus=8 m2m_ml.py \
--domains KDE4,OpenSubtitles,QED,TED2020,Ubuntu,Bible \
--m2m_model facebook/m2m100_418M \
--per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
--log_interval 10 --temp 5.0 --cuda \
--data_path /mounts/data/proj/lavine/multilingual_multidomain/data/original_data/tsv_split \
--save /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/m2m_ml/EU_unseen \
--ds_config /mounts/work/lavine/lavine_code/MMDMT/paper/scripts/ds_config_8_gpus.json \
> /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/m2m_ml/EU_unseen/nohup.txt &