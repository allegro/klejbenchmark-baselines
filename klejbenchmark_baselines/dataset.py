import inspect
import os
import typing as t
from functools import partial

import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from klejbenchmark_baselines.task import BaseTask

Batch = t.Dict[str, torch.Tensor]


class Datasets:

    def __init__(self, task: BaseTask):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=task.config.tokenizer_name_or_path,
            do_lower_case=task.config.do_lower_case,
        )

        self.train_ds = KlejDataset(
            split='train',
            task=task,
            text_encoder=tokenizer,
            target_encoder=None,
        )
        self.valid_ds = KlejDataset(
            split='valid',
            task=task,
            text_encoder=tokenizer,
            target_encoder=self.train_ds.target_encoder,
        )
        self.test_ds = KlejDataset(
            split='test',
            task=task,
            text_encoder=tokenizer,
            target_encoder=self.train_ds.target_encoder,
        )


class KlejDataset(Dataset):

    def __init__(self, split: str, task: BaseTask, text_encoder: PreTrainedTokenizer,
                 target_encoder: t.Optional[BaseEstimator]):

        # config
        self.split = split
        self.task = task

        # load data
        dataset_path = os.path.join(
            self.task.config.task_path,
            getattr(self.task, f'{self.split}_file'),
        )
        raw_data = self._load_data(dataset_path)
        self.parsed_data = self.task.parse_data(raw_data, extract_target=(self.split != 'test'))

        # encoders
        self.text_encoder = text_encoder
        if target_encoder is not None:
            self.target_encoder = target_encoder
        else:
            if self.task.output_type == 'classification':
                self.target_encoder = LabelEncoder().fit(self.parsed_data['target'])
            elif self.task.output_type == 'regression':
                self.target_encoder = FunctionTransformer(
                    func=self._list_as_floats,
                    validate=False,
                ).fit(self.parsed_data['target'])
            else:
                raise KeyError(f'Output type "{self.task.output_type}" is not supported.')

    @staticmethod
    def _load_data(data_path: str) -> pd.DataFrame:
        return pd.read_csv(data_path, sep='\t', quoting=3, skip_blank_lines=False)

    @staticmethod
    def _list_as_floats(lst: t.List[str]) -> t.List[float]:
        return [float(e) for e in lst]

    def __len__(self) -> int:
        return len(self.parsed_data['sentence1'])

    def __getitem__(self, idx: int) -> Batch:

        sentence1 = self.parsed_data['sentence1'][idx]
        if self.parsed_data['sentence2'] is not None:
            sentence2 = self.parsed_data['sentence2'][idx]
        else:
            sentence2 = None

        row = self._encode_text(
            sentence1=sentence1,
            sentence2=sentence2,
            max_len=self.task.config.max_seq_length,
        )

        if 'target' in self.parsed_data:
            row.update(
                self._encode_target(
                    target=self.parsed_data['target'][idx],
                ),
            )

        return row

    def _encode_plus(self, *args, **kwargs) -> t.Dict[str, t.List[float]]:
        """
            Older versions of transformers (e.g. 2.0.0) always return token_type_ids. However,
            this behaviour changed and now (2.8.0) you need to explicitly request for them.
            So we use check if there is argument for returning them and if so, use it.
        """

        encode_func = partial(self.text_encoder.encode_plus, *args, **kwargs)
        encode_args = inspect.getfullargspec(self.text_encoder.encode_plus).args

        if 'return_token_type_ids' in encode_args:
            return encode_func(return_token_type_ids=True)
        else:
            return encode_func()

    def _encode_text(self, sentence1: str, sentence2: t.Optional[str], max_len: int) -> Batch:
        outputs = self._encode_plus(
            text=sentence1,
            text_pair=sentence2,
            add_special_tokens=True,
        )
        seq_len = len(outputs['input_ids'])
        outputs['attention_mask'] = [1] * seq_len

        # truncate
        # warning: it might be incorrect, since we remove special tokens from the end
        outputs['input_ids'] = outputs['input_ids'][:max_len]
        outputs['token_type_ids'] = outputs['token_type_ids'][:max_len]
        outputs['attention_mask'] = outputs['attention_mask'][:max_len]

        # pad to max_len
        pad_len = max_len - seq_len
        pad_id = self.text_encoder.pad_token_id
        outputs['input_ids'] += ([pad_id] * pad_len)
        outputs['token_type_ids'] += ([pad_id] * pad_len)
        outputs['attention_mask'] += ([0] * pad_len)

        # convert to tensors
        output_tensors = {k: torch.tensor(v) for k, v in outputs.items()}

        return output_tensors

    def _encode_target(self, target: str) -> Batch:
        return {'labels': torch.tensor(self.target_encoder.transform([target])[0])}

    def get_dataloader(self, **kwargs) -> t.Iterable[t.Dict]:
        return DataLoader(
            self,
            batch_size=self.task.config.batch_size,
            num_workers=self.task.config.num_workers,
            drop_last=False,
            **kwargs,
        )
