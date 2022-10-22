#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17.12.21 14:30
# @Author  : Wen Lai
# @Site    : 
# @File    : spm_encode.py
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
import subprocess

def main(args):
    ### domain_list
    domain_list = ['EUbookshop', 'KDE4', 'OpenSubtitles', 'QED', 'TED2020', 'Ubuntu']
    ### pre_list
    pre_list = ['mk-sr','et-mk','hr-mk','hr-hu','et-hu','hr-sr','en-sr','hu-sr',
                'hu-mk','en-et','et-sr','en-hr','et-hr','en-hu','en-mk']

    for dl in domain_list:
        os.mkdir(os.path.join(args.output, dl))
        for pre_LL in pre_list:
            print('processing: ' + dl + '.' + pre_LL)
            lang_LL = pre_LL.split('-')

            command = 'python ' + args.spm_script_path \
                      + ' --model ' + args.spm_model_path \
                      + ' --output_format=piece ' \
                      + '--inputs ' + os.path.join(args.input, dl, dl + '.' + pre_LL + '.' + lang_LL[0]) \
                      + ' ' + os.path.join(args.input, dl, dl + '.' + pre_LL + '.' + lang_LL[1]) \
                      + ' --outputs ' + os.path.join(args.output, dl, dl + '.' + pre_LL + '.' + lang_LL[0]) \
                      + ' ' + os.path.join(args.output, dl, dl + '.' + pre_LL + '.' + lang_LL[1])
            subprocess.call(command, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help='input path')
    parser.add_argument("--output", required=True, type=str, help='output path')
    parser.add_argument("--spm_script_path", required=True, type=str, help='spm script path')
    parser.add_argument("--spm_model_path", required=True, type=str, help='spm pretrained model path')
    main(parser.parse_args())