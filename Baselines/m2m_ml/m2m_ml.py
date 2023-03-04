#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 30.03.22 17:55
# @Author  : Wen Lai
# @Site    : 
# @File    : m2m_deepspeed_domain_limit.py
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
import torch, os, time, argparse, logging, pickle, warnings, sys
import numpy as np
from deepspeed.runtime.lr_schedules import WarmupLR
from torch.utils.data import DataLoader, RandomSampler
from data import CorpusM2M100
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist

import deepspeed

from transformers import (
	M2M100Tokenizer,
	SchedulerType,
	MODEL_MAPPING, default_data_collator, DataCollatorForSeq2Seq, M2M100ForConditionalGeneration,
)


logger = logging.getLogger("transformers.tokenization_utils")
warnings.filterwarnings("ignore")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


parser = argparse.ArgumentParser()

parser.add_argument('--meta_lr', type=float, default=5e-5, help='meta learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='')
parser.add_argument('--hidden_dims', type=int, default=768, help='')

parser.add_argument('--task_per_queue', type=int, default=8, help='')
parser.add_argument('--update_step', type=int, default=3, help='')
parser.add_argument('--beta', type=float, default=1.0, help='')
parser.add_argument('--meta_epochs', type=int, default=5, help='iterations')

parser.add_argument('--seed', type=int, default=42, help='seed for numpy and pytorch')
parser.add_argument('--log_interval', type=int, default=200, help='Print after every log_interval batches')
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--save', type=str, default='saved/', help='')
parser.add_argument('--load', type=str, default='', help='')
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--meta_tasks', type=str, default='sc,pa,qa,tc,po')

parser.add_argument('--temp', type=float, default=1.0)
parser.add_argument('--update_dds', type=int,default=10)
parser.add_argument('--dds_lr', type=float,default=0.01)
parser.add_argument('--load_optim_state', action='store_true',help='')
parser.add_argument('--dev_tasks', type=str, default='sc,pa,qa,tc,po')
parser.add_argument('--K', type=int,default=1)

parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_answer_length", default=30, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup", default=0, type=int)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--ds_config", required=True, type=str, help='DS Config Path')

# multi-domain and multilingual machine translation
parser.add_argument('--domains', type=str, default='EUbookshop,KDE4,OpenSubtitles,QED,TED2020,Ubuntu')
parser.add_argument('--data_path', type=str, default='')


parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

parser.add_argument(
	"--predict_with_generate",
	type=bool,
	default=True,
	help="",
)
parser.add_argument(
	"--dataset_config_name",
	type=str,
	default=None,
	help="The configuration name of the dataset to use (via the datasets library).",
)
parser.add_argument(
	"--train_file", type=str, default=None, help="A csv or a json file containing the training data."
)

parser.add_argument(
	"--num_beams",
	type=int,
	default=None,
	help="Number of beams to use for evaluation. This argument will be "
	"passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
)

parser.add_argument(
	"--max_source_length",
	type=int,
	default=128,
	help="The maximum total input sequence length after "
	"tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
	"--max_target_length",
	type=int,
	default=128,
	help="The maximum total sequence length for target text after "
	"tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
	"during ``evaluate`` and ``predict``.",
)
parser.add_argument(
	"--val_max_target_length",
	type=int,
	default=None,
	help="The maximum total sequence length for validation "
	"target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
	"padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
	"param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
)
parser.add_argument(
	"--pad_to_max_length",
	type=bool,
	default=False,
	help="Whether to pad all samples to model maximum sentence "
	"length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
	"efficient on GPU but very bad for TPU.",
)
parser.add_argument(
	"--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
)
parser.add_argument(
	"--ignore_pad_token_for_loss",
	type=bool,
	default=True,
	help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
)
parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
parser.add_argument(
	"--source_prefix",
	type=str,
	default=None,
	help="A prefix to add before every source text " "(useful for T5 models).",
)
parser.add_argument(
	"--preprocessing_num_workers",
	type=int,
	default=None,
	help="The number of processes to use for the preprocessing.",
)
parser.add_argument(
	"--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
)
parser.add_argument(
	"--max_length",
	type=int,
	default=128,
	help=(
		"The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
		" sequences shorter will be padded if `--pad_to_max_lengh` is passed."
	),
)
parser.add_argument(
	"--m2m_model",
	type=str,
	help="Path to pretrained model or model identifier from huggingface.co/models.",
	required=True,
)
parser.add_argument(
	"--config_name",
	type=str,
	default=None,
	help="Pretrained config name or path if not the same as model_name",
)
parser.add_argument(
	"--tokenizer_name",
	type=str,
	default=None,
	help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
	"--use_slow_tokenizer",
	action="store_true",
	help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
)
parser.add_argument(
	"--per_device_train_batch_size",
	type=int,
	default=8,
	help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
	"--per_device_eval_batch_size",
	type=int,
	default=8,
	help="Batch size (per device) for the evaluation dataloader.",
)
parser.add_argument(
	"--learning_rate",
	type=float,
	default=5e-5,
	help="Initial learning rate (after the potential warmup period) to use.",
)
# parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
parser.add_argument(
	"--max_train_steps",
	type=int,
	default=None,
	help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
)
parser.add_argument(
	"--gradient_accumulation_steps",
	type=int,
	default=1,
	help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
	"--lr_scheduler_type",
	type=SchedulerType,
	default="linear",
	help="The scheduler type to use.",
	choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
)
parser.add_argument(
	"--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
)
parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
# parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
parser.add_argument(
	"--model_type",
	type=str,
	default=None,
	help="Model type to use if training from scratch.",
	choices=MODEL_TYPES,
)

parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
parser.add_argument(
	"--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
)
parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")

parser.add_argument("--local_rank", type=int)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#### lang pair list
lang_pair = ['mk-sr', 'et-mk', 'hr-mk', 'hr-hu', 'et-hu',
			 'hr-sr', 'en-sr', 'hu-sr', 'hu-mk', 'en-et',
			 'et-sr', 'en-hr', 'et-hr', 'en-hu', 'en-mk',
			 'sr-mk', 'mk-et', 'mk-hr', 'hu-hr', 'hu-et',
			 'sr-hr', 'sr-en', 'sr-hu', 'mk-hu', 'et-en',
			 'sr-et', 'hr-en', 'hr-et', 'hu-en', 'mk-en'
			 ]

if not os.path.exists(args.save):
	os.makedirs(args.save)

class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		self.log = open(os.path.join(args.save,"output.txt"), "w")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		self.log.flush()

sys.stdout = Logger()
print (args)

if torch.cuda.is_available():
	if not args.cuda:
		args.cuda = True

	torch.cuda.manual_seed_all(args.seed)

task_types = args.domains.split(',')
list_of_tasks = []

for tt in task_types:
	for lp in lang_pair:
		if tt == 'Bible' and 'mk' in lp:
			continue
		else:
			list_of_tasks.append(tt + '_' + lp)

list_of_tasks = list(set(list_of_tasks))
num_tasks = len(list_of_tasks)
print (list_of_tasks)

list_of_dev_tasks = list_of_tasks
dev_task_types = task_types

train_corpus = {}
dev_corpus = {}

## using the tokenizer
tokenizer = M2M100Tokenizer.from_pretrained(args.m2m_model)

for k in list_of_tasks:
	task_list = k.split('_')
	domain = task_list[0]
	lang_list = task_list[1].split('-')
	source_lang = lang_list[0]
	target_lang = lang_list[1]
	train_data_file = os.path.join(args.data_path, task_list[1], domain, 'train.tsv')
	valid_data_file = os.path.join(args.data_path, task_list[1], domain, 'valid.tsv')
	train_corpus[k] = CorpusM2M100(args, tokenizer, train_data_file, domain, source_lang, target_lang, 'train')
	dev_corpus[k] = CorpusM2M100(args, tokenizer, valid_data_file, domain, source_lang, target_lang, 'dev')

model = M2M100ForConditionalGeneration.from_pretrained(args.m2m_model)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        "lr": args.meta_lr
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": args.meta_lr,
    },
]

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=optimizer_grouped_parameters,
    config_params=args.ds_config,
)

### create dataloaders for m2m100 model

train_dataloaders = {}
dev_dataloaders = {}
train_sampler = {}
dev_sampler = {}
psi_train_dataloaders = {}
psi_dev_dataloaders = {}

for k in list_of_tasks:
	if args.pad_to_max_length:
		# If padding was already done ot max length, we use the default data collator that will just convert everything
		# to tensors.
		train_data_collator = default_data_collator
		dev_data_collator = default_data_collator
	else:
		# Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
		# the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
		# of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
		train_data_collator = DataCollatorForSeq2Seq(
			train_corpus[k].tokenizer,
			model=model,
			label_pad_token_id=train_corpus[k].label_pad_token_id,
		)
		dev_data_collator = DataCollatorForSeq2Seq(
			dev_corpus[k].tokenizer,
			model=model,
			label_pad_token_id=dev_corpus[k].label_pad_token_id,
		)

	train_sampler[k] = DistributedSampler(train_corpus[k].datasets['train'])

	train_dataloaders[k] = DataLoader(
		train_corpus[k].datasets['train'],
		collate_fn=train_data_collator,
		batch_size=args.per_device_train_batch_size,
		sampler=train_sampler[k]
	)
	dev_dataloaders[k] = DataLoader(
		dev_corpus[k].datasets,
		collate_fn=dev_data_collator,
		batch_size=args.per_device_eval_batch_size
	)
	psi_train_dataloaders[k] = DataLoader(
		train_corpus[k].datasets['train'],
		collate_fn=train_data_collator,
		batch_size=args.per_device_train_batch_size,
		sampler=RandomSampler(train_corpus[k])
	)
	psi_dev_dataloaders[k] = DataLoader(
		dev_corpus[k].datasets,
		collate_fn=dev_data_collator,
		batch_size=args.per_device_eval_batch_size,
		sampler=RandomSampler(dev_corpus[k])
	)

list_of_psi_train_iters = {k:iter(psi_train_dataloaders[k]) for k in list_of_tasks}
list_of_psi_dev_iters = {k:iter(psi_dev_dataloaders[k]) for k in list_of_tasks}

p = np.array([len(train_dataloaders[y])*1.0/sum([len(train_dataloaders[x]) for x in list_of_tasks]) for y in list_of_tasks])
p_temp = np.power(p, 1.0/args.temp)
p_temp = p_temp / np.sum(p_temp)

psis = {k:torch.log(torch.tensor(p_temp[i])) for i,k in enumerate(list_of_tasks)}

print (psis)

steps = args.meta_epochs * sum([len(train_dataloaders[x]) for x in list_of_tasks]) // (args.task_per_queue * args.update_step)

scheduler = WarmupLR(optimizer, warmup_min_lr=0, warmup_max_lr=3e-5, warmup_num_steps=steps)


logger = {}
logger['total_val_loss'] = []
logger['val_loss'] = {k:[] for k in list_of_tasks}
logger['psis'] = []
logger['train_loss'] = []
logger['args'] = args

def get_batch(dataloader_iter, dataloader):
	try:
		batch = next(dataloader_iter)
	except StopIteration:
		dataloader_iter = iter(dataloader)
		batch = next(dataloader_iter)
	return batch


class Sampler:
	def __init__(self, p, steps, func):
		# Sampling Weights
		self.init_p = p
		self.total_steps = steps
		self.func = func
		self.curr_step = 0

		self.update_step = args.update_step
		self.task_per_queue = args.task_per_queue
		self.list_of_tasks = list_of_tasks
		self.list_of_iters = {k:iter(train_dataloaders[k]) for k in self.list_of_tasks}

	def __iter__(self):
		return self

	def __next__(self):
		curr_p = self.func(self.init_p, self.list_of_tasks, self.curr_step, self.total_steps)
		self.curr_step += 1
		tasks = np.random.choice(self.list_of_tasks,self.task_per_queue,p=curr_p)
		queue = []
		for i in range(self.task_per_queue):
			l = {'task':tasks[i],'data':[]}
			for _ in range(self.update_step):
				l['data'].append(get_batch(self.list_of_iters[tasks[i]], train_dataloaders[tasks[i]]))
			queue.append(l)
		return queue

def identity(x,y,z,w): return x

def UniformBatchSampler():
	p = np.array([len(train_dataloaders[y])*1.0/sum([len(train_dataloaders[x]) for x in list_of_tasks]) for y in list_of_tasks])
	p_temp = np.power(p, 1.0/args.temp)
	p_temp = p_temp / np.sum(p_temp)
	sampler = iter(Sampler(p_temp, steps, identity))
	return sampler

sampler = UniformBatchSampler()
print (sampler.init_p)

def metastep(model, queue):
	t1 = time.time()
	n = len(queue)
	old_vars = []
	running_vars = []
	for param in model.parameters():
		old_vars.append(param.data.cpu().clone())
	losses = [[0 for _ in range(args.update_step)] for _ in range(n)]
	for i in range(n):
		for k in range(args.update_step):
			data_cuda = queue[i]['data'][k].data
			data_cuda_tmp = {}
			data_cuda_tmp['input_ids'] = data_cuda['input_ids'].cuda()
			data_cuda_tmp['attention_mask'] = data_cuda['attention_mask'].cuda()
			data_cuda_tmp['labels'] = data_cuda['labels'].cuda()
			output = model(**data_cuda_tmp)
			loss = output[0].mean()
			torch.cuda.empty_cache()
			model.backward(loss)
			model.step()
			losses[i][k] += loss.item()
			torch.cuda.empty_cache()
		if running_vars == []:
			for _, param in enumerate(model.parameters()):
				running_vars.append(param.data.cpu().clone())
		else:
			for idx, param in enumerate(model.parameters()):
				running_vars[idx].data += param.data.cpu().clone()

		for idx, param in enumerate(model.parameters()):
			param.data = old_vars[idx].data.clone().cuda()
		# torch.cuda.empty_cache()

	for param in running_vars:
		param /= n

	for idx, param in enumerate(model.parameters()):
		param.data = (old_vars[idx].data + args.beta * (running_vars[idx].data - old_vars[idx].data)).cuda()
	# torch.cuda.empty_cache()
	return [(queue[i]['task'],sum(l)/len(l)) for i,l in enumerate(losses)], time.time() - t1


def evaluate(model, task, data):
	with torch.no_grad():
		total_loss = 0.0
		correct = 0.0
		total = 0.0
		for j,batch in enumerate(data):
			data_cuda = batch.data
			data_cuda_tmp = {}
			data_cuda_tmp['input_ids'] = data_cuda['input_ids'].cuda()
			data_cuda_tmp['attention_mask'] = data_cuda['attention_mask'].cuda()
			data_cuda_tmp['labels'] = data_cuda['labels'].cuda()
			output = model.forward(**data_cuda_tmp)
			loss = output[0].mean()
			total_loss += loss.item()
			# loss.cpu()
			# data_cuda_tmp['input_ids'] = data_cuda_tmp['input_ids'].cpu()
			# data_cuda_tmp['attention_mask'] = data_cuda_tmp['attention_mask'].cpu()
			# data_cuda_tmp['labels'] = data_cuda_tmp['labels'].cpu()
			# del loss, data_cuda_tmp
			# torch.cuda.empty_cache()
		total_loss /= len(data)
		return total_loss


def evaluateMeta(model):
	loss_dict = {}
	total_loss = 0
	model.eval()
	for task in list_of_tasks:
		loss = evaluate(model, task, dev_dataloaders[task])
		loss_dict[task] = loss
		total_loss += loss
	return loss_dict, total_loss

def main():

	# Meta learning stage
	print ("*" * 50)
	print ("Meta Learning Stage")
	print ("*" * 50)

	print ('Training for %d metasteps'%steps)

	total_loss = 0

	min_task_losses = 10000

	global_time = time.time()

	try:
		for j,metabatch in enumerate(sampler):
			if j > steps: break
			loss, _ = metastep(model_engine, metabatch)
			total_loss += sum([y for (x,y) in loss])/len(loss)
			logger['train_loss'].append(sum([y for (x,y) in loss])/len(loss))

			if (j + 1) % args.log_interval == 0:
				val_loss_dict, val_loss_total = evaluateMeta(model_engine)
				logger['total_val_loss'].append(val_loss_total)
				for task in val_loss_dict.keys():
					logger['val_loss'][task].append(val_loss_dict[task])

				total_loss /= args.log_interval
				print('Val Loss Dict : ',val_loss_dict)

				loss_per_task = {}
				for task in val_loss_dict.keys():
					if task.split('_')[0] in loss_per_task.keys():
						loss_per_task[task.split('_')[0]] = loss_per_task[task.split('_')[0]] + val_loss_dict[task]
					else:
						loss_per_task[task.split('_')[0]] = val_loss_dict[task]

				print('Time : %f , Step  : %d , Train Loss : %f, Val Loss : %f' % (time.time() - global_time,j+1,total_loss,val_loss_total))
				print('===============================================')
				global_time = time.time()

				avg_all_losses = np.mean(list(loss_per_task.values()))

				# save the model
				if avg_all_losses < min_task_losses:
					min_task_losses = avg_all_losses
					if dist.get_rank() == 0:
						model_engine.module.save_pretrained(os.path.join(args.save, "model"))
						print("Saving Model ... ...")

				total_loss = 0

				with open(os.path.join(args.save, 'logger.pickle'), 'wb') as f:
					pickle.dump(logger, f)
			scheduler.step()
			torch.cuda.empty_cache()
			time.sleep(10)

	except KeyboardInterrupt:
		print ('skipping meta learning')

	with open(os.path.join(args.save,'logger.pickle'),'wb') as f:
		pickle.dump(logger, f)

if __name__ == '__main__':
	main()