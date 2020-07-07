import sys
import typing as t

import numpy as np
import pytorch_lightning as pl
from tqdm.auto import tqdm


class TrainerWithPredictor(pl.Trainer):

    def predict(self) -> t.Dict[str, np.ndarray]:
        self.reset_test_dataloader(self.model)
        max_batches = self.num_test_batches
        if self.fast_dev_run:
            max_batches = 1

        # setup progress bar
        position = 2 * self.process_position + 1
        total = max_batches if max_batches != float('inf') else None
        self.test_progress_bar = tqdm(
            desc='Predicting',
            total=total,
            leave=True,
            position=position,
            disable=not self.progress_bar_refresh_rate,
            dynamic_ncols=True,
            file=sys.stdout,
        )

        # predict
        output = self._evaluate(
            model=self.model,
            dataloaders=self.test_dataloaders,
            max_batches=max_batches,
            test_mode=True,
        )

        return output
