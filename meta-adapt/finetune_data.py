from transformers import M2M100Tokenizer
from torch.utils.data import Dataset
from datasets import load_dataset

class CorpusM2M100(Dataset):
	def __init__(self, args, tokenizer, data_file, domain, source_lang, target_lang, type):
		self.model_name_or_path = args.m2m_model
		self.domain = domain
		self.source_lang = source_lang
		self.target_lang = target_lang
		self.prefix = args.source_prefix if args.source_prefix is not None else ""
		self.max_source_length = args.max_source_length
		self.max_target_length = args.max_target_length
		self.padding = "max_length" if args.pad_to_max_length else False
		self.ignore_pad_token_for_loss = args.ignore_pad_token_for_loss
		self.tokenizer = tokenizer
		self.tokenizer.src_lang = source_lang
		self.tokenizer.tgt_lang = target_lang
		# self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name_or_path, use_fast=not args.use_slow_tokenizer, src_lang=source_lang, tgt_lang=target_lang)
		self.label_pad_token_id = -100 if args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
		self.raw_datasets = self.load_data(data_file)
		self.corpus_len = len(self.raw_datasets['train'])
		self.select_len = int(self.corpus_len * 0.05)
		if type == 'train':
			self.raw_datasets = self.raw_datasets['train'].select(range(int(self.corpus_len * 0.1), self.corpus_len))
			self.column_names = self.raw_datasets.column_names
		else:
			if self.select_len == 0:
				self.raw_datasets = self.raw_datasets['train'].select(range(self.corpus_len))
				self.column_names = self.raw_datasets.column_names
			else:
				self.raw_datasets = self.raw_datasets['train'].select(range(int(self.corpus_len * 0.1)))
				self.column_names = self.raw_datasets.column_names

		self.datasets = self.raw_datasets.map(
			self.preprocess_function,
			batched=True,
			num_proc=args.preprocessing_num_workers,
			remove_columns=self.column_names,
			load_from_cache_file=not args.overwrite_cache,
			desc="Running tokenizer on dataset",
		)

	def preprocess_function(self, examples):
		inputs = [ex for ex in examples[self.source_lang]]
		targets = [ex for ex in examples[self.target_lang]]
		inputs = [self.prefix + inp for inp in inputs]
		model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.padding, truncation=True)
		# Setup the tokenizer for targets
		with self.tokenizer.as_target_tokenizer():
			labels = self.tokenizer(targets, max_length=self.max_target_length, padding=self.padding, truncation=True)

		# If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
		# padding in the loss.
		if self.padding == "max_length" and self.ignore_pad_token_for_loss:
			labels["input_ids"] = [
				[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
			]

		model_inputs["labels"] = labels["input_ids"]
		return model_inputs

	def load_data(self, data_files):
		raw_datasets = load_dataset('csv', data_files=data_files, delimiter='\t')
		return raw_datasets

	def __len__(self):
		return len(self.datasets)

	def __getitem__(self, id):
		return {
				"input_ids": self.datasets['train']['input_ids'][id],
				"attention_mask": self.datasets['train']['attention_mask'][id],
				"labels": self.datasets['train']['labels'][id],
			}
