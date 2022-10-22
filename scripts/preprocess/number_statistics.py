#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11.06.22 20:36
# @Author  : Wen Lai
# @Site    : 
# @File    : number_statistics.py
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
import os.path

import pandas as pd


def main(args):
    # domain_list
    domain_list = ['EUbookshop', 'KDE4', 'OpenSubtitles', 'QED', 'TED2020', 'Ubuntu']
    # pre_list
    lang_list = ['mk-sr', 'et-mk', 'hr-mk', 'hr-hu', 'et-hu',
                 'hr-sr', 'en-sr', 'hu-sr', 'hu-mk', 'en-et',
                 'et-sr', 'en-hr', 'et-hr', 'en-hu', 'en-mk',
                 'sr-mk', 'mk-et', 'mk-hr', 'hu-hr', 'hu-et',
                 'sr-hr', 'sr-en', 'sr-hu', 'mk-hu', 'et-en',
                 'sr-et', 'hr-en', 'hr-et', 'hu-en', 'mk-en',
                 ]
    dict_res = {}
    out_file = open(args.output, 'w', encoding='utf-8')
    for dl in domain_list:
        for ll in lang_list:
            dlp = dl + '-' + ll
            df_tmp = pd.read_csv(os.path.join(args.input, ll, dl, 'train.tsv'), sep='\t')
            if len(df_tmp) != 5000:
                dict_res[dlp] = len(df_tmp)
                out_file.write(dlp + ',' + str(len(df_tmp)) + '\n')

    print(dict_res)
    print(len(dict_res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help='input path')
    parser.add_argument("--output", required=True, type=str, help='output path')
    main(parser.parse_args())
