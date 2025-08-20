# [TCSVT2025] FMTrack: Frequency-aware Interaction and Multi-Expert Fusion for RGB-T Tracking

This is an official pytorch implementation of the 2025 IEEE Transactions on Circuits and Systems for Video Technology paper:
```
FMTrack: Frequency-aware Interaction and Multi-Expert Fusion for RGB-T Tracking
(accepted by IEEE Transactions on Circuits and Systems for Video Technology, DOI: xxx)
```

![image](https://github.com/xyl-507/FMTrack/blob/main/figs/FMTrack.jpg)

The paper can be downloaded from [IEEE Xplore] and [ResearchGate]

The models and raw results can be downloaded from [**[BaiduYun]**](https://pan.baidu.com/s/1pWnuFUtYeuhkWZ7XHfybNQ?pwd=x2w9) and [**[GitHub]**](https://github.com/xyl-507/FMTrack/releases/tag/Results).

The tracking demos are displayed on the [Bilibili](https://www.bilibili.com/video/BV1MoeKzMEdx/).

### Proposed modules
- `FIN` in Line 270 of [vit.py](https://github.com/xyl-507/FMTrack/blob/db0652d35d3ee18af887c0831ee6ac31c1a6e307/lib/models/odtrack/vit_ce_FreqFusion.py#L270)
- `MEFM` in Line 128 of [odtrack.py](https://github.com/xyl-507/FMTrack/blob/db0652d35d3ee18af887c0831ee6ac31c1a6e307/lib/models/odtrack/odtrack.py#L128)

## Requirements
```
pip install -r environment.yml
```

## Results
### RGB-T Tracking

|   RGB-T Datasets (SR/PR)   |      TBSI (TPAMI25)    | FMTrack256 (ours) |
|   --------------------     |   :----------------:   | :---------------: | 
|          RGBT210           |      0.625 / 0.853     |   0.636 / 0.883   |
|          RGBT234           |      0.637 / 0.871     |   0.661 / 0.898   |
|          LasHeR            |      0.556 / 0.692     |   0.576 / 0.727   |
|         VTUAV-ST           |      0.672 / 0.810     |   0.728 / 0.857   |

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```
## Data Preparation
Training datasets download
- LasHeR: [[Link]](https://github.com/BUGPLEASEOUT/LasHeR)
- VTUAV-ST: : [[Link]](https://github.com/zhang-pengyu/DUT-VTUAV)

Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- LasHeR
            |-- test
            |-- train
            ...
        -- VTUAV-ST
            |-- test
            |-- train
   ```

## Training
- Download pre-trained [[MAE ViT-Base weights]](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it to `$PROJECT_ROOT$/FMTrack/lib/models/pretrained_models`.
- Download RGB Tracker pre-trained weights [[ODTrack]](https://github.com/GXNU-ZhongLab/ODTrack) and put it to `$PROJECT_ROOT$/FMTrack/lib/models/pretrained_models`.

1.Training with one GPU.
```
cd /$PROJECT_ROOT$/FMTrack
CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script odtrack --config baseline_256_lasher --save_dir ./output --mode single --nproc_per_node 1
```

2.Training with multiple GPUs.
```
cd /$PROJECT_ROOT$/FMTrack
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script odtrack --config baseline_256_lasher --save_dir ./output --mode multiple --nproc_per_node 2
```

Before training, please make sure the data path in [***local.py***](./lib/train/admin/local.py) is correct.

## Evaluation
Download the model [FMTrack](https://pan.baidu.com/s/1pWnuFUtYeuhkWZ7XHfybNQ?pwd=x2w9), extraction code: `x2w9`. Add the model to `$PROJECT_ROOT$/FMTrack/output/checkpoints/train/`.
```
python tracking/test.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset lasher_test --runid 15 --threads 4 --num_gpus 2
python tracking/analysis_results.py --tracker_name odtrack --tracker_param baseline_256_lasher --dataset_name lasher_test --runid 15
```
- We recommend the official evaluation toolkit for RGBT210, RGBT234, LasHeR, VTUAV !!!.

Before evaluation, please make sure the data path in [***local.py***](./lib/test/evaluation/local.py) is correct.

## Test FLOPs, and Speed
```
python tracking/profile_model.py --script odtrack --config baseline_256_lasher
```

### Acknowledgement
The code based on the [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack),
[FreqFusion](https://github.com/Linwei-Chen/FreqFusion), and [DRSformer](https://github.com/cschenxiang/DRSformer).

We would like to express our sincere thanks to the contributors.

### Citation:
If you find this work useful for your research, please cite the following papers:
```
@ARTICLE{10220112,
  author={Yuanliang Xue,Guodong Jin,Bineng Zhong,Tao Shen,Lining Tan,Chaocan Xue,Yaozong Zheng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={FMTrack: Frequency-aware Interaction and Multi-Expert Fusion for RGB-T Tracking}, 
  year={2025},
  doi={xxx}}
```
If you have any questions about this work, please contact with me via *xyl_507@outlook.com*

