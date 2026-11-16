# CMB-ML: cmbNNCS

This repository implements the cmbNNCS component separation method used within the CMB-ML pipeline, using the neural network implementation provided by [Guo-Jian-Wang/cmbnncs](https://github.com/Guo-Jian-Wang/cmbnncs).

It includes the code and configurations for training, inference, and analysis, as well as instructions for reproducing the cmbNNCS results. This version will reproduce the results from the Direct-to-Power-Spectrum ([D2PS](https://github.com/CMB-ML/e-d2ps)) paper.

## Installation

Follow the installation instructions in the [CMB-ML repository](https://github.com/CMB-ML/cmb-ml), using release `v0.3.0-d2ps`.

With the CMB-ML conda environment activated, clone this repository and install its additional dependencies:

```bash
git clone https://github.com/CMB-ML/cmb-ml-cmbnncs.git
cd cmb-ml-cmbnncs
git switch --detach v0.2.0-d2ps
conda env update -f env.yaml
python -m pip check
```

A compatible NVIDIA GPU and driver are required for GPU execution. The provided environment specifies PyTorch 2.6.0 with CUDA 12.4 and a fixed revision of the upstream cmbNNCS repository.

We have seen small numerical differences in predictions from the same trained model when using different GPU architectures and batch sizes. Reproduction of the reported results was verified using an NVIDIA V100 with an inference batch size of 12.

## Dataset

To reproduce the results reported in the D2PS paper, use the `I_512_1850_UnifWMAP_Wide` dataset, which can be obtained with the dataset download instructions in the CMB-ML repository.

Ensure that the `CMB_ML_DATA` environment variable is set as described in the CMB-ML documentation.

The default configuration, `cfg/config_cmbnncs_unet8.yaml`, is set up to use this dataset.

## Released Model Assets

The trained model used in the D2PS paper is available from [Box](https://utdallas.box.com/s/zrw9om3ddgdmzlkqwlc9imb3jihk44ze).

The archive contains:

- `CMBNNCS_D_Model/model_best.pth`
- `CMBNNCS_D_Model/loss_record.csv`
- `CMBNNCS_B_Extrema/normalization.yaml`

Download the archive using `rclone`:

```bash
mkdir -p cmbNNCS_assets
cd cmbNNCS_assets

rclone copyurl \
    'https://utdallas.app.box.com/index.php?rm=box_download_shared_file&shared_name=zrw9om3ddgdmzlkqwlc9imb3jihk44ze&file_id=f_2590609358166' \
    ./cmbNNCS_UNet8_assets.tar --progress
cd ..
```

The archive has the following properties:

| Property | Value |
|---|---|
| Filename | `cmbNNCS_UNet8_assets.tar` |
| Size | 384.2 MiB (402862080 bytes) |
| SHA-256 | `a6ab610a0bac8b45f1309703196eab12ce918a8ea32cc68b238abfd24d428a73` |

Automatic verification of the checksum can be performed with:
```bash
rclone copyurl \
    'https://utdallas.app.box.com/index.php?rm=box_download_shared_file&shared_name=zrw9om3ddgdmzlkqwlc9imb3jihk44ze&file_id=f_2590625041220' \
    ./cmbNNCS_UNet8_assets.tar.sha256

sha256sum -c cmbNNCS_UNet8_assets.tar.sha256
```

After verifying the checksum, extract the archive into the default working directory for cmbNNCS:

```bash
MODEL_DIR="$CMB_ML_DATA/Datasets/I_512_1850_UnifWMAP_Wide/cmbNNCS_UNet8"

mkdir -p "$MODEL_DIR"
tar -xvf cmbNNCS_UNet8_assets.tar -C "$MODEL_DIR"
```

## Reproducing the Reported Results

The provided model allows reproduction of the reported results without retraining. To do so, in `main_cmbnncs.py`, modify the `pipes` list to include only the following executors:

```python
pipes = [
    PreprocessExecutor,
    PredictionExecutor,
    PostprocessExecutor,
    MaskCreatorExecutor,
    CommonPredPostExecutor,
    MakePredPowerSpectrumExecutor,
]
```

Run the pipeline (only test and validation splits are needed for inference):

```bash
python main_cmbnncs.py 'pipeline.preprocess.splits=[test,valid]'
```

The default configuration uses the provided `best` checkpoint and generates predictions and power spectra for the test and validation splits.

For a smaller test run, the number of simulations processed per split can be limited:

```bash
python main_cmbnncs.py 'pipeline.preprocess.splits=[test,valid]' n_infer_cap=12
```

The default working directory within the dataset is `cmbNNCS_UNet8/`. The resulting power spectra are stored in `cmbNNCS_UNet8/Analysis_K_Power_Spectra/` within the dataset directory. Further post-processing (polynomial fitting) is applied in D2PS.

## Training

To train a new model and generate its predictions, run the complete pipeline using `main_cmbnncs.py` with its original executor list intact.

To avoid overwriting the provided model or its predictions, specify a different working directory:

```bash
python main_cmbnncs.py working_dir=cmbNNCS_UNet8_New/
```

The default configuration uses the UNet8 architecture, a batch size of 12, and 120 training epochs.

Training parameters can be modified in the configuration files or overridden from the command line using Hydra.

## Licensing

- **Source code:** [MIT License](LICENSE).
- **Released model and supporting data assets:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE.data). A copy of the license is also available alongside the downloadable assets on Box.

## Attribution

The cmbNNCS method was proposed by [Guo-Jian Wang, Hong-Liang Shi, Ye-Peng Yan, Jun-Qing Xia, Yan-Yun Zhao, Si-Yu Li, and Jun-Feng Li (2022)](https://doi.org/10.3847/1538-4365/ac5f4a).

If you use this repository in your research, please cite the relevant publications:

- **cmbNNCS method:** [Guo-Jian Wang, Hong-Liang Shi, Ye-Peng Yan, et al. 2022, ApJS, 260, 13](https://doi.org/10.3847/1538-4365/ac5f4a).
- **CMB-ML simulation framework:** [James Amato, Yunan Xie, Leonel Medina-Varela, et al. 2025, ICCV, 9418–9430](https://doi.org/10.1109/ICCV51701.2025.00879).
