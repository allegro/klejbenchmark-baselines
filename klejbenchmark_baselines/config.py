import typing as t
from inspect import getmembers, isdatadescriptor


class BaseConfig:

    @classmethod
    def from_argparse(cls, arguments) -> 'BaseConfig':
        obj = cls()
        for name in obj._prop_names():
            arg_val = arguments.get(name)
            if arg_val is not None:
                setattr(obj, name, arg_val)
        obj.validate_correctness()
        return obj

    @classmethod
    def from_dict(cls, arguments) -> 'BaseConfig':
        obj = cls()
        for name in arguments:
            arg_val = arguments.get(name)
            if arg_val is not None:
                setattr(obj, name, arg_val)
        obj.validate_correctness()
        return obj

    def _prop_names(self) -> t.Set[str]:
        names = set()
        for name, _ in getmembers(type(self), isdatadescriptor):
            if not name.startswith('_'):
                names.add(name)

        for name in self.__dict__.keys():
            if not name.startswith('_'):
                names.add(name)

        return names

    def validate_correctness(self):
        raise NotImplementedError


class Config(BaseConfig):

    def __init__(self):
        super().__init__()

        # Task
        self.task_name: t.Optional[str] = None
        self.run_id: t.Optional[str] = None

        # Data
        self.task_path: t.Optional[str] = None
        self.predict_path: t.Optional[str] = None
        self.logger_path: t.Optional[str] = None
        self.checkpoint_path: t.Optional[str] = None

        # Tokenizer
        self.tokenizer_name_or_path: t.Optional[str] = None
        self.max_seq_length: int = 256
        self.do_lower_case: bool = False

        # Model
        self.model_name_or_path: t.Optional[str] = None
        self.learning_rate: float = 2e-5
        self.adam_epsilon: float = 1e-8
        self.warmup_steps: int = 100
        self.batch_size: int = 16
        self.gradient_accumulation_steps: int = 2
        self.num_train_epochs: int = 4
        self.weight_decay: float = 0.0
        self.max_grad_norm: float = 1.0

        # Other
        self.seed: int = 42
        self.num_workers: int = 4
        self.num_gpu: int = 1

    def validate_correctness(self):
        assert self.num_train_epochs > 0
