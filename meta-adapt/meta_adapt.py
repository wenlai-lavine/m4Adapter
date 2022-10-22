#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 24.04.22 22:03
# @Author  : Wen Lai
# @Site    : 
# @File    : finetune_single_task.py
# @Usage information: 

# Copyright (c) 2021-present, CIS, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Example:

nohup python meta_adapt.py \
--device_id 6 \
--domain EUbookshop \
--finetune_scripts /mounts/work/lavine/lavine_code/MMDMT/paper/finetune_meta_adapter_unseen.py \
--generate_scripts /mounts/work/lavine/lavine_code/MMDMT/paper_repro/unseen/generate_base_adapter.py \
--data_path /mounts/data/proj/lavine/multilingual_multidomain/data/original_data/tsv_split \
--adapter_path /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_adapters/EU_unseen/save_adapter \
--save_path /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_adapters/EU_unseen/unseen \
--generate_path /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_adapters/EU_unseen/unseen/generate_results \
> /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_adapters/EU_unseen/unseen/log.txt &

nohup python meta_adapt.py \
--device_id 1 \
--domain EUbookshop \
--finetune_scripts /mounts/work/lavine/lavine_code/MMDMT/paper/finetune_meta_adapter_unseen.py \
--generate_scripts /mounts/work/lavine/lavine_code/MMDMT/paper_repro/unseen/generate_base_adapter.py \
--data_path /mounts/data/proj/lavine/multilingual_multidomain/data/original_data/tsv_split \
--adapter_path /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_pre_adapters/EU_unseen/save_adapter \
--save_path /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_pre_adapters/EU_unseen/unseen \
--generate_path /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_pre_adapters/EU_unseen/unseen/generate_results \
> /mounts/data/proj/lavine/multilingual_multidomain/experiments/paper_repro_new/Baselines/meta_pre_adapters/EU_unseen/unseen/log.txt &
"""
import argparse
import os
import random
import subprocess


def main(args):
    lang_list = ['mk-sr', 'et-mk', 'hr-mk', 'hr-hu', 'et-hu',
                 'hr-sr', 'en-sr', 'hu-sr', 'hu-mk', 'en-et',
                 'et-sr', 'en-hr', 'et-hr', 'en-hu', 'en-mk',
                 'sr-mk', 'mk-et', 'mk-hr', 'hu-hr', 'hu-et',
                 'sr-hr', 'sr-en', 'sr-hu', 'mk-hu', 'et-en',
                 'sr-et', 'hr-en', 'hr-et', 'hu-en', 'mk-en',
                 ]
    for lang in lang_list:
        meta_task = args.domain + '_' + lang
        os.makedirs(os.path.join(args.generate_path, lang), exist_ok=True)
        finetune_cmd = 'CUDA_VISIBLE_DEVICES=' + args.device_id \
                       + ' python -m torch.distributed.launch --nproc_per_node=1 --master_port ' + str(random.randint(0, 100000)) + ' ' + args.finetune_scripts \
                       + ' --meta_epochs 5 --task_per_queue 1 --meta_task ' + meta_task \
                       + ' --adapter_type pfeiffer --m2m_model facebook/m2m100_418M' \
                       + ' --per_device_train_batch_size 8 --per_device_eval_batch_size 8' \
                       + ' --log_interval 2 --temp 5.0' \
                       + ' --data_path ' + os.path.join(args.data_path, lang, args.domain) \
                       + ' --adapter_path ' + args.adapter_path \
                       + ' --save ' + args.save_path
        generate_cmd = 'CUDA_VISIBLE_DEVICES=' + args.device_id \
                       + ' python ' + args.generate_scripts \
                       + ' --adapter_type pfeiffer' \
                       + ' --lang_pair ' + lang \
                       + ' --domain ' + args.domain \
                       + ' --input_path ' + args.data_path \
                       + ' --output_path ' + args.generate_path \
                       + ' --adapter_path ' + args.save_path

        subprocess.call(finetune_cmd, shell=True)
        print('finish finetuning ' + lang + ' ... ...')
        subprocess.call(generate_cmd, shell=True)
        print('finish generate ' + lang + ' ... ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", required=True, type=str, help='')
    parser.add_argument("--finetune_scripts", required=True, type=str, help='')
    parser.add_argument("--generate_scripts", required=True, type=str, help='')
    parser.add_argument("--domain", required=True, type=str, help='')
    parser.add_argument("--data_path", required=True, type=str, help='')
    parser.add_argument("--adapter_path", required=True, type=str, help='')
    parser.add_argument("--save_path", required=True, type=str, help='')
    parser.add_argument("--generate_path", required=True, type=str, help='')
    main(parser.parse_args())