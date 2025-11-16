"""
This script runs a pipeline for prediction and analysis of the cleaned CMB signal using CMBNNCS.

The pipeline consists of the following steps:
1. Preprocessing the data
2. Training the model
3. Predicting the cleaned CMB signal
4. Postprocessing the predictions
5. Converting predictions to common form for comparison across models
6. Generating per-pixel analysis results for each simulation
7. Generating per-pixel summary statistics for each simulation
8. Converting the theory power spectrum to a format that can be used for analysis
9. Generating per-ell power spectrum analysis results for each simulation
10. Generating per-ell power spectrum summary statistics for each simulation

And also generating various analysis figures, throughout.

Final comparison is performed in the main_analysis_compare.py script.

Usage:
    python main_cmbnncs.py
"""
import logging

import hydra

from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims import MaskCreatorExecutor
from cmbnncs_local import (
                           HydraConfigcmbNNCSCheckerExecutor,
                           PreprocessMakeScaleExecutor,
                           PreprocessExecutor,
                        #    NonParallelPreprocessExecutor,
                           TrainingExecutor,
                           PredictionExecutor,
                           PostprocessExecutor,
                        #    NonParallelPostprocessExecutor
                           cmbNNCSMakePSExecutor,
                           cmbNNCSShowPostExecutor
                           )
from cmbml.analysis import (
    CommonRealPostExecutor,
    CommonPredPostExecutor,
    LossPlotExecutor
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_cmbnncs_unet8")
def cmbnncs(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    pipes = [
        # HydraConfigCheckerExecutor,
        # HydraConfigCMBNNCSCheckerExecutor,
        # PreprocessMakeScaleExecutor,
        # PreprocessExecutor,
        TrainingExecutor,
        PredictionExecutor,
        PostprocessExecutor,
        cmbNNCSMakePSExecutor,
        CommonRealPostExecutor,
        CommonPredPostExecutor,
        cmbNNCSShowPostExecutor,
        LossPlotExecutor
    ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_cmbnncs_op")
def cmbnncs_on_planck(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    pipes = [
        PreprocessExecutor,
        PredictionExecutor,
        PostprocessExecutor,
        cmbNNCSMakePSExecutor,
        cmbNNCSShowPostExecutor
    ]
    run(cfg, pipes)


def run(cfg, pipes):
    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)
    for pipe in pipes:
        pipeline_context.add_pipe(pipe)

    pipeline_context.prerun_pipeline()
    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        logger.exception("An exception occured during the pipeline.", exc_info=e)
        raise e
    finally:
        logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    cmbnncs()
    # cmbnncs_on_planck()
