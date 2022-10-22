#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17.12.21 14:14
# @Author  : Wen Lai
# @Site    : 
# @File    : match_str.py
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
import re

from tqdm import tqdm


def main(args):
    ### domain_list
    domain_list = ['EUbookshop', 'KDE4', 'OpenSubtitles', 'QED', 'TED2020', 'Ubuntu']

    ### pre_list
    pre_list = ['mk-sr', 'et-mk', 'hr-mk', 'hr-hu', 'et-hu', 'hr-sr', 'en-sr', 'hu-sr',
                'hu-mk', 'en-et', 'et-sr', 'en-hr', 'et-hr', 'en-hu', 'en-mk']

    for dl in domain_list:
        os.mkdir(os.path.join(args.output, dl))
        for pre_LL in pre_list:
            print('process ' + pre_LL + ' corpus ... ...')
            lang_LL = pre_LL.split('-')
            count = 0

            src_out = open(os.path.join(args.output, dl, dl + '.' + pre_LL + '.' + lang_LL[0]), 'wt', encoding=args.encoding)
            tgt_out = open(os.path.join(args.output, dl, dl + '.' + pre_LL + '.' + lang_LL[1]), 'wt', encoding=args.encoding)
            with open(os.path.join(args.input, dl, dl + '.' + pre_LL + '.' + lang_LL[0]), 'rt', encoding=args.encoding) as f_src:
                with open(os.path.join(args.input, dl, dl + '.' + pre_LL + '.' + lang_LL[1]), 'rt', encoding=args.encoding) as f_tgt:
                    for src_line, tgt_line in tqdm(zip(f_src, f_tgt)):
                        flag_s = re.search(r'</?\w+[^>]*>', src_line)
                        flag_t = re.search(r'</?\w+[^>]*>', tgt_line)
                        if flag_s or flag_t:
                            count += 1
                            continue
                        else:
                            src_out.write(re.sub(r'</?\w+[^>]*>', ' ', src_line).strip() + '\n')
                            tgt_out.write(re.sub(r'</?\w+[^>]*>', ' ', tgt_line).strip() + '\n')
            print(pre_LL + ' corpus delete ' + str(count) + ' sentence ... ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help='input path', default='')
    parser.add_argument("--output", type=str, help='input path', default='')
    parser.add_argument('--encoding', default='utf-8', help='character encoding for input/output')
    main(parser.parse_args())