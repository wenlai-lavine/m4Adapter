#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13.05.22 10:28
# @Author  : Wen Lai
# @Site    : 
# @File    : generate_results.py
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
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AdapterConfig


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
    ### domain_list
    domain_list = args.domain.split(',')
    lang_list = ['mk-sr', 'et-mk', 'hr-mk', 'hr-hu', 'et-hu',
                 'hr-sr', 'en-sr', 'hu-sr', 'hu-mk', 'en-et',
                 'et-sr', 'en-hr', 'et-hr', 'en-hu', 'en-mk',
                 'sr-mk', 'mk-et', 'mk-hr', 'hu-hr', 'hu-et',
                 'sr-hr', 'sr-en', 'sr-hu', 'mk-hu', 'et-en',
                 'sr-et', 'hr-en', 'hr-et', 'hu-en', 'mk-en',
                 ]

    model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_adapter(
        os.path.join(args.adapter_path),
        config='pfeiffer',
        load_as='agonostic_adapters',
    )
    model.set_active_adapters(['agonostic_adapters'])
    model.to(device)

    for lang in lang_list:
        src_lang = lang.split('-')[0]
        tgt_lang = lang.split('-')[1]
        for dl in domain_list:
            if dl == 'Bible' and 'mk' in lang:
                continue
            print('generate ' + lang + ' in ' + dl + ' domain.')
            os.makedirs(os.path.join(args.output_path, lang, dl))
            src_list_total = tsv2list(os.path.join(args.input_path, lang, dl, 'test.tsv'), src_lang)
            tgt_decode_list = []
            for src_list in tqdm(src_list_total):
                tokenizer.src_lang = src_lang
                encode_src = tokenizer(src_list, truncation=True, padding=True, return_tensors="pt")
                encode_src = encode_src.to(device)
                generate_tgt_tokens = model.generate(**encode_src, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
                tgt_decode_list.append(tokenizer.batch_decode(generate_tgt_tokens, skip_special_tokens=True))

            tgt_res_list = [b for a in tgt_decode_list for b in a]
            list2txt(tgt_res_list, os.path.join(args.output_path, lang, dl, 'generate_' + lang + '.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, type=str, help='')
    parser.add_argument("--input_path", required=True, type=str, help='')
    parser.add_argument("--adapter_path", required=True, type=str, help='')
    parser.add_argument("--output_path", required=True, type=str, help='')
    main(parser.parse_args())