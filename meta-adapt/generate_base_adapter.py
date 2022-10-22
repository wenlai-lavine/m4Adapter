#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 26.04.22 14:48
# @Author  : Wen Lai
# @Site    : 
# @File    : generate_base.py
# @Usage information: 

# Copyright (c) 2021-present, CIS, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Example: 

"""
import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def text2list(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            result.append(line.strip())
    return result

def tsv2list(file_path, src_lang):
    tsv_file = pd.read_csv(file_path, sep='\t')
    results_total = tsv_file[src_lang].to_list()
    results = [results_total[i:i + 8] for i in range(0, len(results_total), 8)]
    return results

def list2txt(decode_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as decode_txt:
        for dl in decode_list:
            decode_txt.write(dl.strip() + '\n')

def main(args):
    src_lang = args.lang_pair.split('-')[0]
    tgt_lang = args.lang_pair.split('-')[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
    model.load_adapter(
        os.path.join(args.adapter_path),
        config=args.adapter_type,
        load_as='meta-agnostic-adapter',
    )
    model.set_active_adapters(['meta-agnostic-adapter'])

    model.to(device)
    os.makedirs(os.path.join(args.output_path, args.lang_pair), exist_ok=True)
    src_list_total = tsv2list(os.path.join(args.input_path, args.lang_pair, args.domain, 'test.tsv'), src_lang)
    tgt_decode_list = []
    for src_list in tqdm(src_list_total):
        tokenizer.src_lang = src_lang
        encode_src = tokenizer(src_list, truncation=True, padding=True, return_tensors="pt")
        encode_src = encode_src.to(device)
        generate_tgt_tokens = model.generate(**encode_src, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        tgt_decode_list.append(tokenizer.batch_decode(generate_tgt_tokens, skip_special_tokens=True))

    tgt_res_list = [b for a in tgt_decode_list for b in a]
    list2txt(tgt_res_list, os.path.join(args.output_path, args.lang_pair, 'generate_' + args.lang_pair + '.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, type=str, help='')
    parser.add_argument("--lang_pair", required=True, type=str, help='')
    parser.add_argument("--input_path", required=True, type=str, help='')
    parser.add_argument("--adapter_path", required=True, type=str, help='')
    parser.add_argument("--adapter_type", required=True, type=str, help='')
    parser.add_argument("--output_path", required=True, type=str, help='')
    main(parser.parse_args())