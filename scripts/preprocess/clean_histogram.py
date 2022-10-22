#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17.12.21 11:13
# @Author  : Wen Lai
# @Site    : 
# @File    : clean_histogram.py
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

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Path to input')
parser.add_argument('--output', type=str, help='Path to output')
parser.add_argument('--encoding', default='utf-8', help='character encoding for input/output')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
parser.add_argument('--threshold-character', type=str, default=']', help='Threshold character')
parser.add_argument('--histograms', type=str, help='Path to histograms')

args = parser.parse_args()

def read_hist(f):
    ch = []
    for line in f:
        c = line[0]
        if c == args.threshold_character:
            break
        ch.append(c)
    return ch

### domain_list
domain_list = ['EUbookshop', 'KDE4', 'OpenSubtitles', 'QED', 'TED2020', 'Ubuntu']

### pre_list
pre_list = ['mk-sr','et-mk','hr-mk','hr-hu','et-hu','hr-sr','en-sr',
            'hu-sr','hu-mk','en-et','et-sr','en-hr','et-hr','en-hu','en-mk']


for dl in domain_list:
    os.mkdir(os.path.join(args.output, dl))
    for pre_LL in pre_list:
        print('processing' + pre_LL + '.....')
        lang_LL = pre_LL.split('-')
        remain_count = 0
        del_count = 0

        ## accept characters filtering
        with(open("{}/{}".format(args.histograms, lang_LL[0]), 'r', encoding='utf8')) as f:
            ch1 = read_hist(f)

        with(open("{}/{}".format(args.histograms, lang_LL[1]), 'r', encoding='utf8')) as f:
            ch2 = read_hist(f)

        print("Accepted characters for {}: {}".format(lang_LL[0], len(ch1)))
        print("Accepted characters for {}: {}".format(lang_LL[1], len(ch2)))

        src_out = open(os.path.join(args.output, dl, dl + '.' + pre_LL + '.' + lang_LL[0]), 'wt', encoding=args.encoding)
        tgt_out = open(os.path.join(args.output, dl, dl + '.' + pre_LL + '.' + lang_LL[1]), 'wt', encoding=args.encoding)
        with open(os.path.join(args.input, dl, dl + '.' + pre_LL + '.' + lang_LL[0]), 'rt', encoding=args.encoding) as f_src:
            with open(os.path.join(args.input, dl, dl + '.' + pre_LL + '.' + lang_LL[1]), 'rt', encoding=args.encoding) as f_tgt:
                for ori_1, ori_2 in tqdm(zip(f_src, f_tgt)):
                    src, tgt = ori_1.strip(), ori_2.strip()

                    cnt1 = len([c for c in src if c in ch1])
                    cnt2 = len([c for c in tgt if c in ch2])

                    if cnt1 / len(src) > args.threshold and cnt2 / len(tgt) > args.threshold:
                        src_out.write(src + '\n')
                        tgt_out.write(tgt + '\n')
                        remain_count += 1
                    else:
                        del_count += 1

        print(pre_LL + ' number of delete sentences is ' + str(del_count))
        print(pre_LL + ' number of remain sentences is ' + str(remain_count))