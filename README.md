# The KLEJ Benchmark Baselines
The [KLEJ benchmark](https://klejbenchmark.com/) (Kompleksowa Lista Ewaluacji Językowych) is a set of nine evaluation tasks for the Polish language understanding.

This repository contains example scripts to easily fine-tune models from the [transformers](https://github.com/huggingface/transformers) library on the KLEJ benchmark.

## Installation
Install the Python package using the following commands:
```bash
$ git clone https://github.com/allegro/klejbenchmark-baselines
$ pip install klejbenchmark-baselines/
```

## Quick Start
To fine-tune your model on KLEJ tasks using the default settings, you can use the provided example scripts.

First, download the KLEJ benchmark datasets:
```bash
$ bash scripts/download_klej.sh
```

After downloading KLEJ, customize training parameters inside the `scripts/run_training.sh` script and train the models using:
```bash
$ bash scripts/run_training.sh
```

It will create:
- Tensorboard logs with training and validation metrics,
- checkpoints of the best models,
- a zip file with predictions for the test sets, which is a valid submission for the KLEJ benchmark.

The zip file can be submitted at the [klejbenchmark.com](https://klejbenchmark.com/submit/) website for the evaluation on the test sets.

## Custom Training
It's also possible to train each model separately and customize the training parameters using the `klejbenchmark_baselines/main.py` script.

## License
Apache 2 License

## Citation
If you use this code, please cite the following paper:

```
@inproceedings{rybak-etal-2020-klej,
    title = "{KLEJ}: Comprehensive Benchmark for Polish Language Understanding",
    author = "Rybak, Piotr and Mroczkowski, Robert and Tracz, Janusz and Gawlik, Ireneusz",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.111",
    pages = "1191--1201",
}
```

## Authors
This code was created by the **Allegro Machine Learning Research** team.

You can contact us at: <a href="mailto:klejbenchmark@allegro.pl">klejbenchmark@allegro.pl</a>
