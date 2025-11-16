from omegaconf import DictConfig
from cmbml.analysis import ShowSimsPostExecutor


class cmbNNCSShowPostExecutor(ShowSimsPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        stage_str = "show_cmb_post_masked"
        super().__init__(cfg, stage_str)
        self.right_subplot_title = "cmbNNCS Predicted"
        self.suptitle = cfg.fig_model_name
