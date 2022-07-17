# D2ADA: Dynamic Density-aware Active Domain Adaptation for Semantic Segmentation

Official pytorch implementation of ["D2ADA: Dynamic Density-aware Active Domain Adaptation for Semantic Segmentation. Wu et al. ECCV 2022."](https://arxiv.org/abs/2202.06484).

In this work, we present D2ADA, a general active domain adaptation framework for domain adaptive semantic segmentation. Here is a brief introduction video of our work (Remember to turn on the sound :grinning:).

https://user-images.githubusercontent.com/22555914/179380681-020ac953-4538-49c3-8438-c8068c353b26.mov

## Environmental Setup

- OS: Ubuntu 20.04
- CUDA: 11.3
- Installation
  ```
  conda env create -f environment.yml
  ```

## A. Data Preparation

- Download [Cityscapes](https://www.cityscapes-dataset.com), [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/), and [SYNTHIA](https://synthia-dataset.net) datasets.

- Region division (For training only): run the following preprocessing code for the three datasets.
  ```
  python3 data_preprocessing/superpixel_gen.py \
      --dataset {cityscapes,GTA5,SYNTHIA} --datadir <DATASETDIR>
  ```
  - Note: For SYNTHIA dataset, the `--datadir` argument should be `<SYNTHIA-ROOT>/RAND_CITYSCAPES/`.

## B. Training

### Step0: Model Warm-up

For simplicity, you can download our pretrained warm-up models directly. As an alternative, you can run the following scripts for UDA warm-up (or supervision warm-up) yourself.

| Model                   | Benchmark | mIoU  | Download |
| ----------------------- | --------- | ----- | -------- |
| DeepLabV2-ResNet101     | GTA5      | 44.61 | [Link](https://drive.google.com/file/d/1m9h1TWdoUQJeMvP3Xu7yqYZ4q_jpQAJT/view?usp=sharing)     |
| DeepLabV3Plus-ResNet101 | GTA5      | 45.51 | [Link](https://drive.google.com/file/d/1P2eEYq3lSo8bRXHnUM3oJe4fa05fnVUk/view?usp=sharing)    |
| DeepLabV2-ResNet101     | SYNTHIA   | 39.95 | [Link](https://drive.google.com/file/d/17tO4wMsdw8NSs-yh2FTzuL8ZuY_g6XiF/view?usp=sharing)     |
| DeepLabV3Plus-ResNet101 | SYNTHIA   | 43.04 | [Link](https://drive.google.com/file/d/1FXCBdUDEFIls7DAn8NPVmKzlAKNj1sij/view?usp=sharing)     |

<details><summary>How to run our warm-up script</summary><p>

```bash
# Model Warm-up 
CUDA_VISIBLE_DEVICES=X python3 warmup.py -p <exp-path> [--warmup {uda_warmup, sup_warmup}]
```
- Default arguments are configured in `utils/common.py`. You can modify it via input parameters.
    - `-m`: Choose model backbone, like deeplabv2\_resnet101 or deeplabv3plus\_resnet101.
    - `--src_dataset`: Choose GTA5 or SYNTHIA dataset.
    - `--src_data_dir, --trg_data_dir, --val_data_dir`: Set dataset path.
    - `--src_datalist`: Use either GTA5 datalist or SYNTHIA datalist.
    
</p></details>

### Step1: Run our D2ADA framework

```bash
CUDA_VISIBLE_DEVICES=X python3 train_ADA.py -p <exp-path> --init_checkpoint <checkpoint path> \
    --save_feat_dir <feature directory> [--datalist_path PREVIOUS_DATALIST_PATH] 
```

- Default arguments are configured in `utils/common.py`. You can modify it via input parameters. 
  - `--init_checkpoint`: Specify the path of initial model. You can use our warm-up model in iteration \#0 or use `exp-path/checkpoint0X.tar` to continue the experiment at iteration \#X.
  - `save_feat_dir`: directory to save information (GMM models, region fearures, ...) when conducting density-aware selection.
  - `--datalist_path`: Load previous datdalist to continue the experiment.

<details><summary>Other Notes</summary><p>

- In addition to our D2ADA active learning method, we provide a number of unorganized code of [active learning baselines](./active_selection/other_AL_baselines). If you are interested in the topic, feel free to modify these code for further research.

- Known Issue: We found that bugs were occasionally triggered when programs [constructing density estimators (GMMs)]((https://github.com/tsunghan-wu/itri_project_final/blob/feat/release/active_selection/density_aware_selection.py#L140)). Specifically, the forked program comes out to perform density estimator will not be terminated, and the main process will always wait for the subprocess. As a workaround in our experiment, when this happens, we always press CTRL-C and load the checkpoint and datalist from the previous round to continue execution. If you know how to fix this, please let me know or send a pull request.

</p></details>

## C. Testing (Demo) and Evaluation

### Step0: Download Pretrained Models

Here we provide a number of pretrained models along with selected region lists.

- `checkpoint00.tar`: Initial model (our wram-up model)
- `checkpoint01.tar ~ checkpoint05.tar`: ADA model with 1% target annotation, ... ADA model with 5% target annotation.

| Model                   | Benchmark |  Download |
| ----------------------- | --------- | ----- | 
| DeepLabV2-ResNet101     | GTA5      |  [Link](https://drive.google.com/drive/folders/1mFIiay1NNtHnVMF76FyMR5LbLuQWJy13?usp=sharing)     |
| DeepLabV3Plus-ResNet101 | GTA5      |  [Link](https://drive.google.com/drive/folders/18EtDczRG1wgeSM0A3zq_-rxnsFQESDD5?usp=sharing)     |
| DeepLabV2-ResNet101     | SYNTHIA   |  [Link](https://drive.google.com/drive/folders/1WuO48_c4vmLYEl64gmU7wwnwVmDd-2KQ?usp=sharing)     |
| DeepLabV3Plus-ResNet101 | SYNTHIA   |  [Link](https://drive.google.com/drive/folders/1HD7hORne_5QZKYwz1HlVdKEq_ZlzfyRi?usp=sharing)     |

<details><summary>Ways to analyze selected regions</summary><p>

`datalist_0X.pkl` contains the information about the current labeled training set (including the original GTA5/SYNTHIA dataset and incrementally selected cityscapes regions). You can use the following example script to view or analyze our selected regions for further investigation or future research.

```python3
import pickle
fname = "datalist_01.pkl"
with open(fname, "rb") as f:
    data = pickle.load(f)
# dict_keys(['src_label_im_idx', 'trg_label_im_idx', 'trg_pool_im_idx', 'trg_label_suppix', 'trg_pool_suppix'])
# The data structure is the same as "dataloader/region_active_dataset.py"
``` 
</p></details>

### Step1: Predict Semantic Labeling

```bash
python3 inference.py --trained_model_path <trained_model_path> [--save_dir INFERENCE_RESULT_DIR]
```

### Step2: Compute mIoU

```bash
python3 evaluation.py --root_dir <Cityscapes data root> --pred_dir <saved predicted directory>
```

## Citation
```
@article{wu2022d2ada,
  title={D2ADA: Dynamic Density-aware Active Domain Adaptation for Semantic Segmentation},
  author={Wu, Tsung-Han and Liou, Yi-Syuan and Yuan, Shao-Ji and Lee, Hsin-Ying and Chen, Tung-I and Huang, Kuan-Chih and Hsu, Winston H},
  journal={arXiv preprint arXiv:2202.06484},
  year={2022}
}
```
