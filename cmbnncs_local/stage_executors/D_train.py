import logging

from tqdm import tqdm

# import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from .pytorch_model_base_executor import BaseCMBNNCSModelExecutor
from cmbml.core import Split, Asset

from cmbml.core.asset_handlers import (
    NumpyMap,
    Config, 
    AppendingCsvHandler
    )

from cmbml.torch.pytorch_model_handler import PyTorchModel
# from core.pytorch_dataset import TrainCMBMapDataset
from cmbnncs_local.dataset import TrainCMBMapDataset
# from cmbml.core.pytorch_transform import TrainToTensor
# from cmbml.cmbnncs_local.preprocessing.scale_methods_factory import get_scale_class
# from cmbml.cmbnncs_local.preprocessing.transform_pixel_rearrange import sphere2rect


logger = logging.getLogger(__name__)


class TrainingExecutor(BaseCMBNNCSModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train")

        self.out_model: Asset = self.assets_out["model"]
        self.out_loss_record: Asset = self.assets_out["loss_record"]
        out_model_handler: PyTorchModel
        out_loss_record: AppendingCsvHandler

        self.in_model: Asset = self.assets_in["model"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        # self.in_norm: Asset = self.assets_in["dataset_stats"]  # TODO: Does removing this line break anything?
        in_model_handler: PyTorchModel
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap
        in_norm_handler: Config

        self.norm_data = None

        model_precision = cfg.model.network.model_precision
        self.dtype = self.dtype_mapping[model_precision]
        self.choose_device(cfg.model.train.device)

        self.num_workers = cfg.model.train.num_loader_workers
        self.lr_init = cfg.model.train.learning_rate
        self.lr_final = cfg.model.train.learning_rate_min
        self.repeat_n = cfg.model.train.repeat_n
        self.n_epochs = cfg.model.train.n_epochs
        self.batch_size = cfg.model.train.batch_size
        self.checkpoint = cfg.model.train.checkpoint_every
        self.extra_check = cfg.model.train.extra_check
        self.earliest_best = cfg.model.train.earliest_best
        # self.scale_class = None
        # self.set_scale_class(cfg)

        self.restart_epoch = cfg.model.train.restart_epoch

    # def set_scale_class(self, cfg):
    #     scale_method = cfg.model.preprocess.scaling
    #     self.scale_class = get_scale_class(method=scale_method, 
    #                                        dataset="train", 
    #                                        scale="scale")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        dets_str = ', '.join([str(k) for k in self.instrument.dets.keys()])
        logger.info(f"Creating model using detectors: {dets_str}")

        logger.info(f"Using exponential learning rate scheduler.")
        logger.info(f"Initial learning rate is {self.lr_init}")
        logger.info(f"Final minimum learning rate is {self.lr_final}")
        logger.info(f"Number of epochs is {self.n_epochs}")
        logger.info(f"Batch size is {self.batch_size}")
        logger.info(f"Checkpoint every {self.checkpoint} iterations")
        logger.info(f"Extra check is set to {self.extra_check}")

        train_split = self.splits[0]
        valid_split = self.splits[1]

        # template_split = self.splits[0]
        # dataset = self.set_up_dataset(template_split)
        # train_dataloader = DataLoader(
        #     dataset, 
        #     batch_size=self.batch_size, 
        #     shuffle=True,
        #     )

        train_dataset = self.set_up_dataset(train_split)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
            )

        valid_dataset = self.set_up_dataset(valid_split)
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
            )

        model = self.make_model().to(self.device)

        lr_init = self.lr_init
        lr_final = self.lr_final
        loss_function = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

        # Match CMBNNCS's updates per batch, (not the more standard per epoch)
        total_iterations = self.n_epochs * len(train_dataloader)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda iteration: (lr_final / lr_init) ** (iteration / total_iterations))

        if self.restart_epoch is not None:
            logger.info(f"Restarting training at epoch {self.restart_epoch}")
            # The following returns the epoch number stored in the checkpoint 
            #     as well as loading the model and optimizer with checkpoint information
            with self.name_tracker.set_context("epoch", self.restart_epoch):
                start_epoch = self.in_model.read(model=model, 
                                                 epoch=self.restart_epoch, 
                                                 optimizer=optimizer, 
                                                 scheduler=scheduler)
            if start_epoch == "init":
                start_epoch = 0
            self.out_loss_record.start()  # New lines will be added, naively 
                                          #  Naively = appended; if resuming, manual cleaning may be needed.
        else:
            logger.info(f"Starting new model.")
            with self.name_tracker.set_context("epoch", "init"):
                self.out_model.write(model=model, epoch="init")
            start_epoch = 0
            loss_record_headers = ['Epoch', 'Training Loss', 'Validation Loss']
            self.out_loss_record.write(data=loss_record_headers)

        n_epoch_digits = len(str(self.n_epochs))

        all_train_loss = []
        all_valid_loss = []

        for epoch in range(start_epoch, self.n_epochs):
            train_loss = 0.0
            batch_n = 0
            batch_loss = 0
            with tqdm(train_dataloader, desc="Training", postfix={'Loss': 0}) as pbar:
                for train_features, train_label in pbar:
                    batch_n += 1

                    train_features = train_features.to(device=self.device, dtype=self.dtype)
                    train_label = train_label.to(device=self.device, dtype=self.dtype)

                    # Repeating the training for each batch three times. 
                    # This is strange, but it's what CMBNNCS does.
                    # If implementing a new model, this is not recommended.
                    for _ in range(self.repeat_n):
                        optimizer.zero_grad()
                        output = model(train_features)
                        loss = loss_function(output, train_label)
                        loss.backward()
                        optimizer.step()
                        batch_loss += loss.item()

                    scheduler.step()
                    pbar.set_postfix({'Loss': loss.item() / self.batch_size})

                    batch_loss = batch_loss / self.repeat_n
                    train_loss += batch_loss

            train_loss /= len(train_dataloader.dataset)
            all_train_loss.append(train_loss)
            logger.info(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {train_loss:.4f}')

            model.eval()
            valid_loss = 0
            with torch.no_grad():
                with tqdm(train_dataloader, desc="Validating", postfix={'Loss': 0}) as pbar:
                    for valid_features, valid_label in pbar:
                        valid_features = valid_features.to(device=self.device, dtype=self.dtype)
                        valid_label = valid_label.to(device=self.device, dtype=self.dtype)
                        output = model(valid_features)
                        valid_loss += loss_function(output, valid_label).item()
                    valid_loss /= len(valid_dataloader)
                    all_valid_loss.append(valid_loss)
                    note_min_loss = " *" if valid_loss == min(all_valid_loss) else ""
                    logger.info(f"Epoch {epoch + 1:<{n_epoch_digits}} Validation loss: {valid_loss:.02e}{note_min_loss}")

                    self.out_loss_record.append([epoch + 1, train_loss, valid_loss])

            is_best_condition_a = valid_loss == min(all_valid_loss)
            is_best_condition_b = epoch >= self.earliest_best
            if is_best_condition_a and is_best_condition_b:
                with self.name_tracker.set_context("epoch", "best"):
                    self.out_model.write(model=model,
                                        epoch="best"
                                        )

            # Checkpoint every so many epochs
            condition_a = (epoch + 1) in self.extra_check
            condition_b = (epoch + 1) % self.checkpoint == 0
            if (condition_a or condition_b):
                with self.name_tracker.set_context("epoch", epoch + 1):
                    self.out_model.write(model=model,
                                         optimizer=optimizer,
                                         epoch=epoch + 1)

        with self.name_tracker.set_context("epoch", "final"):
            self.out_model.write(model=model, epoch="final")

            # # Checkpoint every so many epochs
            # if (epoch + 1) in self.extra_check or (epoch + 1) % self.checkpoint == 0:
            #     with self.name_tracker.set_context("epoch", epoch + 1):
            #         self.out_model.write(model=model,
            #                              optimizer=optimizer,
            #                              scheduler=scheduler,
            #                              epoch=epoch + 1,
            #                              loss=epoch_train_loss)

            # is_best_condition_a = epoch_train_loss == min(all_valid_loss)
            # is_best_condition_b = epoch >= self.earliest_best
            # if is_best_condition_a and is_best_condition_b:
            #     with self.name_tracker.set_context("epoch", "best"):
            #         self.out_model.write(model=model,
            #                             epoch="best"
            #                             )


    def set_up_dataset(self, template_split: Split) -> None:
        cmb_path_template = self.make_fn_template(template_split, self.in_cmb_asset)
        obs_path_template = self.make_fn_template(template_split, self.in_obs_assets)

        dataset = TrainCMBMapDataset(
            n_sims = template_split.n_sims,
            freqs = self.instrument.dets.keys(),
            map_fields=self.map_fields,
            label_path_template=cmb_path_template, 
            label_handler=NumpyMap(),
            feature_path_template=obs_path_template,
            feature_handler=NumpyMap()
            )
        return dataset

    def inspect_data(self, dataloader):
        train_features, train_labels = next(iter(dataloader))
        logger.info(f"{self.__class__.__name__}.inspect_data() Feature batch shape: {train_features.size()}")
        logger.info(f"{self.__class__.__name__}.inspect_data() Labels batch shape: {train_labels.size()}")
        npix_data = train_features.size()[-1] * train_features.size()[-2]
        npix_cfg  = hp.nside2npix(self.nside)
        assert npix_cfg == npix_data, "Npix for loaded map does not match configuration yamls."
