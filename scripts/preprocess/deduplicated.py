#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17.12.21 10:43
# @Author  : Wen Lai
# @Site    : 
# @File    : deduplicated.py
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="src file")
    parser.add_argument("--output", type=str, required=True, help="tgt file")
    parser.add_argument('--encoding', default='utf-8', help='character encoding for input/output')
    args = parser.parse_args()

    ### domain_list
    domain_list = ['EUbookshop', 'KDE4', 'OpenSubtitles', 'QED', 'TED2020', 'Ubuntu']

    ### pre_list
    pre_list = ['mk-sr', 'et-mk', 'hr-mk', 'hr-hu', 'et-hu', 'hr-sr', 'en-sr', 'hu-sr',
                'hu-mk', 'en-et', 'et-sr', 'en-hr', 'et-hr', 'en-hu', 'en-mk']
    for dl in domain_list:
        os.mkdir(os.path.join(args.output, dl))
        for pre_LL in pre_list:
            seen = set()
            dup_count = 0
            print('processing: ' + dl + '----' + pre_LL)
            lang_LL = pre_LL.split('-')

            src_out = open(os.path.join(args.output, dl, dl + '.' + pre_LL + '.' + lang_LL[0]), 'wt', encoding=args.encoding)
            tgt_out = open(os.path.join(args.output, dl, dl + '.' + pre_LL + '.' + lang_LL[1]), 'wt', encoding=args.encoding)
            with open(os.path.join(args.input, dl, dl + '.' + pre_LL + '.' + lang_LL[0]), 'rt', encoding=args.encoding) as f_src:
                with open(os.path.join(args.input, dl, dl + '.' + pre_LL + '.' + lang_LL[1]), 'rt', encoding=args.encoding) as f_tgt:
                    for ori_1, ori_2 in tqdm(zip(f_src, f_tgt)):
                        if (ori_1.strip(), ori_2.strip()) not in seen:
                            src_out.write(ori_1.strip() + '\n')
                            tgt_out.write(ori_2.strip() + '\n')
                            seen.add((ori_1.strip(), ori_2.strip()))
                        else:
                            dup_count += 1
                    print(pre_LL + 'file number of duplication: ' + str(dup_count))

if __name__ == "__main__":
    main()