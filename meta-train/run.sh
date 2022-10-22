#!/bin/bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_$USER
nohup python -m torch.distributed.launch --nproc_per_node=8 m2m_meta_adapter.py \
--meta_epochs 10 \
--domains EUbookshop,OpenSubtitles,QED,TED2020,Ubuntu,Bible \
--m2m_model facebook/m2m100_418M \
--per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
--log_interval 10 --temp 5.0 --cuda \
--data_path /mounts/data/proj/lavine/multilingual_multidomain/data/original_data/tsv_split \
--save /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_adapters/KDE_unseen \
> /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_adapters/KDE_unseen/log.txt &