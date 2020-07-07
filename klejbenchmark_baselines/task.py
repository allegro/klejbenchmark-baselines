import typing as t

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score

from klejbenchmark_baselines.config import Config
from klejbenchmark_baselines.metrics import weighted_mean_absolute_error


class BaseTask:

    output_type = None
    train_file = None
    valid_file = None
    test_file = None

    def __init__(self, config: Config):
        self.config = config

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        raise NotImplementedError

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        raise NotImplementedError

    def parse_data(self, raw_data: pd.DataFrame,
                   extract_target: bool = True) -> t.Dict[str, np.ndarray]:
        output = dict()
        output['sentence1'] = self.extract_sentence1(raw_data).astype(dtype=np.unicode_)

        sent2 = self.extract_sentence2(raw_data)
        output['sentence2'] = sent2.astype(dtype=np.unicode_) if sent2 is not None else None

        if extract_target:
            output['target'] = self.extract_target(raw_data).astype(dtype=np.unicode_)

        return output


class NKJPNERTask(BaseTask):

    output_type = 'classification'
    train_file = 'train.tsv'
    valid_file = 'dev.tsv'
    test_file = 'test_features.tsv'

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['sentence'].values

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        return None

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['target'].values

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        return {
            'accuracy': accuracy_score(true, pred),
        }


class CDSCETask(BaseTask):

    output_type = 'classification'
    train_file = 'train.tsv'
    valid_file = 'dev.tsv'
    test_file = 'test_features.tsv'

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['sentence_A'].values

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        return raw_data['sentence_B'].values

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['entailment_judgment'].values

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        return {
            'accuracy': accuracy_score(true, pred),
        }


class CDSCRTask(BaseTask):

    output_type = 'regression'
    train_file = 'train.tsv'
    valid_file = 'dev.tsv'
    test_file = 'test_features.tsv'

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['sentence_A'].values

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        return raw_data['sentence_B'].values

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['relatedness_score'].values

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        return {
            'spearman': spearmanr(a=pred, b=true)[0],
        }


class CBDTask(BaseTask):

    output_type = 'classification'
    train_file = 'train.tsv'
    valid_file = 'train.tsv'
    test_file = 'test_features.tsv'

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['sentence'].values

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        return None

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['target'].values

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        return {
            'f1_score': f1_score(true, pred),
        }


class PolEmoInTask(BaseTask):

    output_type = 'classification'
    train_file = 'train.tsv'
    valid_file = 'dev.tsv'
    test_file = 'test_features.tsv'

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['sentence'].values

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        return None

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['target'].values

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        return {
            'accuracy': accuracy_score(true, pred),
        }


class PolEmoOutTask(BaseTask):

    output_type = 'classification'
    train_file = 'train.tsv'
    valid_file = 'dev.tsv'
    test_file = 'test_features.tsv'

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['sentence'].values

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        return None

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['target'].values

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        return {
            'accuracy': accuracy_score(true, pred),
        }


class DYKTask(BaseTask):

    output_type = 'classification'
    train_file = 'train.tsv'
    valid_file = 'train.tsv'
    test_file = 'test_features.tsv'

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['question'].values

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        return raw_data['answer'].values

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['target'].values

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        return {
            'f1_score': f1_score(true, pred),
        }


class PSCTask(BaseTask):

    output_type = 'classification'
    train_file = 'train.tsv'
    valid_file = 'train.tsv'
    test_file = 'test_features.tsv'

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['extract_text'].values

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        return raw_data['summary_text'].values

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['label'].values

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        return {
            'f1_score': f1_score(true, pred),
        }


class ARTask(BaseTask):

    output_type = 'regression'
    train_file = 'train.tsv'
    valid_file = 'dev.tsv'
    test_file = 'test_features.tsv'

    def extract_sentence1(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['text'].values

    def extract_sentence2(self, raw_data: pd.DataFrame) -> t.Optional[np.ndarray]:
        return None

    def extract_target(self, raw_data: pd.DataFrame) -> np.ndarray:
        return raw_data['rating'].values

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> t.Dict[str, float]:
        return {
            'wmae': 1 - weighted_mean_absolute_error(true, pred)/4.0,
        }


TASKS = {
    'nkjp-ner': NKJPNERTask,
    'cdsc-e': CDSCETask,
    'cdsc-r': CDSCRTask,
    'cbd': CBDTask,
    'polemo2.0-in': PolEmoInTask,
    'polemo2.0-out': PolEmoOutTask,
    'dyk': DYKTask,
    'psc': PSCTask,
    'ar': ARTask,
}
