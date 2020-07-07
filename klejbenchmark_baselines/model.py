import typing as t

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from transformers import AdamW, AutoConfig, AutoModelForSequenceClassification

from klejbenchmark_baselines.dataset import Datasets
from klejbenchmark_baselines.task import BaseTask

Batch = t.Dict[str, torch.Tensor]


class KlejTransformer(pl.LightningModule):

    MODELS_REQUIRING_TOKEN_TYPE_IDS = [
        'albert', 'bert', 'distilbert', 'xlnet',
    ]

    def __init__(self, task: BaseTask, datasets: Datasets):
        super().__init__()

        self.task = task
        self.datasets = datasets

        if task.output_type == 'classification':
            num_labels = len(self.datasets.train_ds.target_encoder.classes_)
        else:
            num_labels = 1

        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.task.config.model_name_or_path,
            num_labels=num_labels,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.task.config.model_name_or_path,
            config=self.config,
        )
        self.lr_scheduler = None

    def forward(self, **inputs) -> t.Dict[str, torch.Tensor]:
        return self.model(**inputs)

    def _get_optimizer(self) -> Optimizer:
        no_decay = ['bias', 'LayerNorm.weight']
        decay_params = [
            p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        no_decay_params = [
            p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.task.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.task.config.learning_rate,
            eps=self.task.config.adam_epsilon,
        )

        return optimizer

    def _get_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        batches_per_epoch = len(self.datasets.train_ds) // self.task.config.batch_size
        iters_per_epoch = batches_per_epoch // max(1, self.task.config.num_gpu)
        steps_per_epoch = iters_per_epoch // self.task.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * float(self.task.config.num_train_epochs)

        def lr_lambda(step: int) -> float:
            if self.task.config.warmup_steps > 0 and step < self.task.config.warmup_steps:
                return step / self.task.config.warmup_steps
            else:
                return (total_steps - step) / (total_steps - self.task.config.warmup_steps)

        lr_scheduler = LambdaLR(optimizer, lr_lambda)

        return lr_scheduler

    def configure_optimizers(self) -> Optimizer:
        optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_scheduler(optimizer)
        return optimizer

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()

    def _prepare_batch(self, batch: Batch) -> Batch:
        if self.model.config.model_type not in self.MODELS_REQUIRING_TOKEN_TYPE_IDS:
            del batch['token_type_ids']

        return batch

    def train_dataloader(self) -> t.Iterable[t.Dict]:
        return self.datasets.train_ds.get_dataloader(
            shuffle=True,
        )

    def training_step(self, batch: Batch, batch_idx: int) -> t.Dict[str, t.Any]:
        model_outputs = self(**self._prepare_batch(batch))
        loss = model_outputs[0]

        metrics = {
            'lr': self.lr_scheduler.get_last_lr()[-1],
            'train_loss': loss.detach().cpu().numpy(),
        }

        output = {
            'loss': loss,
            'log': metrics,
            'progress_bar': metrics,
        }

        return output

    def val_dataloader(self) -> t.Iterable[t.Dict]:
        return self.datasets.valid_ds.get_dataloader(
            shuffle=False,
        )

    def validation_step(self, batch: Batch, batch_idx: int) -> t.Dict[str, t.Any]:
        model_outputs = self(**self._prepare_batch(batch))

        output = {
            'val_loss': model_outputs[0].detach().cpu().numpy(),
            'val_true': batch['labels'].detach().cpu().numpy(),
        }

        logits = model_outputs[1].detach().cpu().numpy()
        if self.task.output_type == 'classification':
            output['val_pred'] = np.argmax(logits, axis=1)
        elif self.task.output_type == 'regression':
            output['val_pred'] = np.squeeze(logits)
        else:
            raise KeyError(f'Output type "{self.task.output_type}" is not supported.')

        return output

    def validation_epoch_end(self, outputs: t.List[t.Dict[str, t.Any]]) -> t.Dict[str, t.Any]:
        loss = np.array([x['val_loss'] for x in outputs])
        pred = np.concatenate([x['val_pred'] for x in outputs])
        true = np.concatenate([x['val_true'] for x in outputs])

        metrics = {
            'val_loss': np.mean(loss),
            **{f'val_{key}': val for (key, val) in self.task.compute_metrics(true, pred).items()},
        }

        output = {
            'log': metrics,
            'progress_bar': metrics,
        }

        return output

    def test_dataloader(self) -> t.Iterable[t.Dict]:
        return self.datasets.test_ds.get_dataloader(
            shuffle=False,
        )

    def test_step(self, batch: Batch, batch_idx: int) -> t.Dict[str, t.Any]:
        model_outputs = self(**self._prepare_batch(batch))

        logits = model_outputs[0].detach().cpu().numpy()
        if self.task.output_type == 'classification':
            pred_ids = np.argmax(logits, axis=1)
            pred = self.datasets.test_ds.target_encoder.inverse_transform(pred_ids)
        elif self.task.output_type == 'regression':
            pred = np.squeeze(logits)
        else:
            raise KeyError(f'Output type "{self.task.output_type}" is not supported.')

        return {'labels': pred}

    def test_epoch_end(self, outputs: t.List[t.Dict[str, t.Any]]) -> t.Dict[str, t.Any]:
        pred = np.concatenate([x['labels'] for x in outputs])

        return {'labels': pred}
