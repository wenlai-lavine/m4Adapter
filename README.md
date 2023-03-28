# $m^4Adapter$
This repository is used in our paper:

$m^4Adapter$: Multilingual Multi-Domain Adaptation for Machine Translation with a Meta-Adapter (Findings of EMNLP 2022)
[paper link](https://arxiv.org/abs/2210.11912)

Please cite our paper and kindly give a star for this repository if you use this code.

------

**Requirements**

1. torch==1.8.1
2. transformers
3. adapter-transformers
4. deepspeed==0.6.0

------

#### Updates (2023-03-28): upload the code for using adapter in m2m_100 model. please copy the ```adapters/m2m.py``` to the ```adapter-transformer``` path

------

#### Step 1: preprocess

+ Download the original corpus from [OPUS](https://opus.nlpl.eu/), contains different domains (*EUbookshop, KDE, OpenSubtitles, QED, TED, Ubuntu, Bible, UN, Tanzil, infopankki*) and different language (*en, de, fr, mk, sr, et, hr, hu, fi, uk, is, lt, ar, es, ru, zh*).
+ Limit the *training dataset* to 5000 samples in each domain and language pair and the *adapting dataset* to 500 samples in each domain and language pair, because there are some language pair without 5000 sentences in the specific domain (see more details in *Appendix A.1*).
+ most of the preprocess scripts are in ```scripts/preprocess```

------

#### Step 2: Baselines

running baseline systems in ```Baselines```



#### Step 3: Meta-Train

```python
## run scripts in meta-train
deepspeed --num_gpus=8 main_deepspeed.py \
--domains EUbookshop,KDE4,OpenSubtitles,QED,TED2020,Ubuntu \
--model_name_or_path facebook/m2m100_418M \
--data_path YOUR_DATA_PATH \
--per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
--cuda --log_interval 20 --temp 5.0
```

------

#### Step 4: Meta-Adapt

```python
## run scripts in meta-adapt
nohup python meta_adapt.py \
--device_id 6 \
--domain EUbookshop \
--finetune_scripts finetune_meta_adapter_unseen.py \
--generate_scripts generate_base_adapter.py \
--data_path YOUR_DATA_PATH \
--adapter_path ADAPTER_PATH \
--save_path MODEL_PATH \
--generate_path GENERATE_PATH \
> log.txt &
```

------

#### Citation

```bibtex
@inproceedings{lai-etal-2022-4,
    title = "m{\textasciicircum}4 Adapter: Multilingual Multi-Domain Adaptation for Machine Translation with a Meta-Adapter",
    author = "Lai, Wen  and
      Chronopoulou, Alexandra  and
      Fraser, Alexander",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.315",
    pages = "4282--4296",
    abstract = "Multilingual neural machine translation models (MNMT) yield state-of-the-art performance when evaluated on data from a domain and language pair seen at training time. However, when a MNMT model is used to translate under domain shift or to a new language pair, performance drops dramatically. We consider a very challenging scenario: adapting the MNMT model both to a new domain and to a new language pair at the same time. In this paper, we propose m{\textasciicircum}4Adapter (Multilingual Multi-Domain Adaptation for Machine Translation with a Meta-Adapter), which combines domain and language knowledge using meta-learning with adapters. We present results showing that our approach is a parameter-efficient solution which effectively adapts a model to both a new language pair and a new domain, while outperforming other adapter methods. An ablation study also shows that our approach more effectively transfers domain knowledge across different languages and language information across different domains.",
}
```

