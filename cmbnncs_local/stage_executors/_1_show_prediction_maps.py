from omegaconf import DictConfig
from cmbml.analysis import CommonRealPostExecutor


class cmbNNCSShowPostExecutor(CommonRealPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "cmbNNCS Predicted"
        self.suptitle = cfg.fig_model_name
