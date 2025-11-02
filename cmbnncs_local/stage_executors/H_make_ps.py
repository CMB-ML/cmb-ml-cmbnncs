from omegaconf import DictConfig

from cmbml.analysis import MakePredPowerSpectrumExecutor


class CMBNNCSMakePSExecutor(MakePredPowerSpectrumExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, "beam_cmbnncs")
