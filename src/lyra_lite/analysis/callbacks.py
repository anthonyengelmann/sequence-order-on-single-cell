from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from lyra_lite.analysis.multiseed import aggregate_sweep


class AggregateCallback(Callback):
    """After a multirun sweep finishes, automatically aggregate its metrics."""

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        sweep_dir = config.hydra.sweep.dir
        aggregate_sweep(sweep_dir)