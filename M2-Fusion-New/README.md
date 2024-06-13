# M2-Fusion
This is a repo of [M2-Fusion](https://ieeexplore.ieee.org/abstract/document/9991894) for 3D object detection.

The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
<!-- 
![image](https://github.com/Link2Link/FE_GCN/blob/main/fig/full_stureture2.png)
![image](https://github.com/Link2Link/FE_GCN/blob/main/fig/figure_gt_pp_fe.png) -->

## Introduction
Multi-modal fusion plays a critical role in 3D object detection, overcoming the inherent limitations of single-sensor perception in autonomous driving. Most fusion methods require data from high-resolution cameras and LiDAR sensors, which are less robust and the detection accuracy drops drastically with the increase of range as the point cloud density decreases. Alternatively, the fusion of Radar and LiDAR alleviates these issues but is still a developing field, especially for 4D Radar with a more robust and broader detection range. Therefore, we are the first to propose a novel fusion method termed M2-Fusion for 4D Radar and LiDAR, based on Multi-modal and Multi-scale fusion.To better integrate two sensors, we propose an Interaction-based Multi-Modal Fusion (IMMF) method utilizing a self-attention mechanism to learn features from each modality and exchange intermediate layer information. Specific to the precision and efficiency balance problem of the current single resolution voxel division, we also put forward a Center-based Multi-Scale Fusion (CMSF) method to regress the center points of objects first and extract features in multiple resolutions. Furthermore, we present a data preprocessing method based on Gaussian distribution that effectively decreases data noise to reduce errors caused by point cloud divergence of 4D Radar data in the x-z plane. A series of experiments were conducted using the Astyx HiRes 2019 dataset, including the calibrated 4D Radar and 16-line LiDAR data, to evaluate the proposed fusion method. The results demonstrated that our fusion method compared favorably with state-of-the-art algorithms, producing mAP (mean average precision) increases of 5.64% and 13.57% for 3D and BEV (bird’s eye view) detection of the car class at a moderate level, respectively.

* Model Framework:
<p align="center">
  <img src="doc/1672290326253.jpg" width="95%">
</p>

## Environment
该代码主要基于OpenPCDet.

- Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04
- Python 3.6+
- PyTorch 1.1 or higher (本项目PyTorch 1.10)
- CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)（本项目基于cuda11.3+PyTorch1.10）
- [spconv v1.0 (commit 8da6f96)](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [spconv v1.2]((https://github.com/traveller59/spconv/tree/v1.2.1)) or [spconv v2.x](https://github.com/traveller59/spconv)（本项目基于spconv1.2.1）

### Installation

a. Clone this repository
```shell
https://github.com/adept-thu/M2-Fusion-New.git
```

b. Install the dependent libraries as follows

* Install the dependent python libraries:
```
pip install -r requirements.txt 
```
c. Prepare the dataset

```
M2_Fusion
├── data
│ ├── mine
│ │ │──lidar
│ ││ │── ImageSets
│ ││ │── training
│ ││ │ ├──calib & velodyne & label_2 & image_2 & arbe
│ ││ │── testing
│ ││ │ ├──calib & velodyne & image_2
│ ├── astyx
├── pcdet
│ ├── datasets
│ ├── models
│ │ │──backbones_2d
│ │ │──backbones_3d
│ │ │──dense_heads
│ │ │──detectors
│ │ │──model_utils
│ ├── ops
│ ├── utils
├── tools
│ ├── cfgs
│ ├── eval_utils
│ ├── scripts
│ ├── test.py 
│ ├── train.py
	
```

d. Generate dataloader

```
#创建astyx数据集
python -m pcdet.datasets.astyx.astyx_dataset create_astyx_infos tools/cfgs/dataset_configs/astyx_dataset.yaml

#创建Dual-Radar数据集
python -m pcdet.datasets.mine.kitti_dataset create_mine_infos tools/cfgs/dataset_configs/mine_dataset.yaml
注：运行上面指令前请将pcdet/datasets/processor/data_processor.py代码中77-82行注释掉，运行完之后再将代码恢复回来。
```

### Train & Evaluation

- To train the model on single GPU, prepare the total dataset and run

```
python train.py --cfg_file ${CONFIG_FILE}
#python train.py --cfg_file cfgs/astyx_models/pointpillar.yaml --extra_tag yourmodelname
```

- To evaluate the model on single GPU, modify the path and run

```
python test.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}
#python test.py --cfg_file cfgs/astyx_models/pointpillar.yaml --ckpt astyx_models/pointpillar/debug/ckpt/checkpoint_epoch_80.pth
```
## Experiment Results:

* All experiments are tested on Astyx Hires2019
<div align=center>
 <table>
 <colgroup>
  <col width="69" style="width:52pt"> 
  <col width="78" style="mso-width-source:userset;mso-width-alt:2496;width:59pt"> 
  <col width="69" span="6" style="width:52pt"> 
  <col width="69" style="width:52pt"> 
 </colgroup>
 <tbody>
  <tr height="19" align=center> 
   <td rowspan="3" class="xl65">Modality</td> 
   <td rowspan="3" class="xl65">Methods</td> 
   <td rowspan="3" class="xl65">Reference</td> 
   <td colspan="3" class="xl65">3D mAP(%)</td> 
   <td colspan="3" class="xl65">BEV mAP(%)</td> 
  </tr> 
  <tr height="19" align=center> 
   <td rowspan="2" class="xl65">Easy</td> 
   <td rowspan="2" class="xl65">Moderate</td> 
   <td rowspan="2" class="xl65">Hard</td> 
   <td rowspan="2" class="xl65">Easy</td> 
   <td rowspan="2" class="xl65">Moderate</td> 
   <td rowspan="2" class="xl65">Hard</td> 
  </tr> 
  <tr height="19" align=center> 
  </tr> 
  <tr height="19" align=center> 
   <td rowspan="7" class="xl65">4D Radar</td> 
   <td class="xl65">PointRCNN</td> 
   <td class="xl65">CVPR 2019</td> 
   <td class="xl65">14.79</td> 
   <td class="xl65">11.4</td> 
   <td class="xl65">11.32</td> 
   <td class="xl65">26.71</td> 
   <td class="xl65">18.74</td> 
   <td class="xl65">18.6</td> 
  </tr> 
  <tr height="28" align=center> 
   <td class="xl65">SECOND</td> 
   <td class="xl65">SENSORS 2018</td> 
   <td class="xl65">23.26</td> 
   <td class="xl65">18.02</td> 
   <td class="xl65">17.06</td> 
   <td class="xl65">37.92</td> 
   <td class="xl65">31.01</td> 
   <td class="xl65">28.83</td> 
  </tr> 
  <tr height="19" align=center> 
   <td class="xl65">PV-RCNN</td> 
   <td class="xl65">CVPR 2020</td> 
   <td class="xl65">27.61</td> 
   <td class="xl65">22.08</td> 
   <td class="xl65">20.51</td> 
   <td class="xl65">49.17</td> 
   <td class="xl65">39.88</td> 
   <td class="xl65">36.5</td> 
  </tr> 
  <tr height="19" align=center> 
   <td rowspan="2" class="xl65">PointPillars</td> 
   <td rowspan="2" class="xl65">CVPR 2019</td> 
   <td rowspan="2" class="xl65">26.03</td> 
   <td rowspan="2" class="xl65">20.49</td> 
   <td rowspan="2" class="xl65">20.4</td> 
   <td rowspan="2" class="xl65">47.38</td> 
   <td rowspan="2" class="xl65">38.21</td> 
   <td rowspan="2" class="xl65">36.74</td> 
  </tr> 
  <tr height="19"> 
  </tr> 
  <tr height="21" align=center> 
   <td class="xl65">Part-A2</td> 
   <td class="xl65">TPAMI 2021</td> 
   <td class="xl65">14.96</td> 
   <td class="xl65">13.76</td> 
   <td class="xl65">13.17</td> 
   <td class="xl65">26.46</td> 
   <td class="xl65">21.47</td> 
   <td class="xl65">20.98</td> 
  </tr> 
  <tr height="19" align=center> 
   <td class="xl65">Voxel R-CNN</td> 
   <td class="xl65">AAAI 2021</td> 
   <td class="xl65">23.65</td> 
   <td class="xl65">18.71</td> 
   <td class="xl65">18.47</td> 
   <td class="xl65">37.77</td> 
   <td class="xl65">31.26</td> 
   <td class="xl65">27.83</td> 
  </tr> 
  <tr height="19" align=center> 
   <td rowspan="7" class="xl65">16-Line LiDAR</td> 
   <td class="xl65">PointRCNN</td> 
   <td class="xl65">CVPR 2019</td> 
   <td class="xl65">39.03</td> 
   <td class="xl65">29.97</td> 
   <td class="xl65">29.66</td> 
   <td class="xl65">41.34</td> 
   <td class="xl65">34.22</td> 
   <td class="xl65">32.95</td> 
  </tr> 
  <tr height="28" align=center> 
   <td class="xl65">SECOND</td> 
   <td class="xl65">SENSORS 2018</td> 
   <td class="xl65">51.75</td> 
   <td class="xl65">43.54</td> 
   <td class="xl65">40.72</td> 
   <td class="xl65">55.16</td> 
   <td class="xl65">45.63</td> 
   <td class="xl65">43.57</td> 
  </tr> 
  <tr height="19" align=center> 
   <td class="xl65">PV-RCNN</td> 
   <td class="xl65">CVPR 2020</td> 
   <td class="xl65">54.63</td> 
   <td class="xl65">44.71</td> 
   <td class="xl65">41.26</td> 
   <td class="xl65">56.08</td> 
   <td class="xl65">46.68</td> 
   <td class="xl65">44.86</td> 
  </tr> 
  <tr height="19" align=center> 
   <td rowspan="2" class="xl65">PointPillars</td> 
   <td rowspan="2" class="xl65">CVPR 2019</td> 
   <td rowspan="2" class="xl65">54.37</td> 
   <td rowspan="2" class="xl65">44.21</td> 
   <td rowspan="2" class="xl65">41.81</td> 
   <td rowspan="2" class="xl65">58.64</td> 
   <td rowspan="2" class="xl65">47.67</td> 
   <td rowspan="2" class="xl65">45.26</td> 
  </tr> 
  <tr height="19"> 
  </tr> 
  <tr height="21" align=center> 
   <td class="xl65">Part-A2</td> 
   <td class="xl65">TPAMI 2021</td> 
   <td class="xl65">45.41</td> 
   <td class="xl65">38.45</td> 
   <td class="xl65">36.74</td> 
   <td class="xl65">49.85</td> 
   <td class="xl65">41.85</td> 
   <td class="xl65">38.93</td> 
  </tr> 
  <tr height="19" align=center> 
   <td class="xl65">Voxel R-CNN</td> 
   <td class="xl65">AAAI 2021</td> 
   <td class="xl65">52.26</td> 
   <td class="xl65">44.08</td> 
   <td class="xl65">40.06</td> 
   <td class="xl65">53.94</td> 
   <td class="xl65">44.54</td> 
   <td class="xl65">40.43</td> 
  </tr> 
  <tr height="29" align=center> 
   <td class="xl65">Camera + 4D Radar</td> 
   <td class="xl65">MVX-Net</td> 
   <td class="xl65">ICRA 2019</td> 
   <td class="xl65">13.2</td> 
   <td class="xl65">11.69</td> 
   <td class="xl65">11.43</td> 
   <td class="xl65">23.57</td> 
   <td class="xl65">20.36</td> 
   <td class="xl65">19.04</td> 
  </tr> 
  <tr height="29" align=center> 
   <td class="xl65">Camera + 16-Line LiDAR</td> 
   <td class="xl65">MVX-Net</td> 
   <td class="xl65">ICRA 2019</td> 
   <td class="xl65">39.16</td> 
   <td class="xl65">31.43</td> 
   <td class="xl65">30.4</td> 
   <td class="xl65">47.04</td> 
   <td class="xl65">38.15</td> 
   <td class="xl65">35.6</td> 
  </tr> 
  <tr height="29" align=center> 
   <td class="xl65">4D Radar + 16-Line LiDAR</td> 
   <td class="xl65">M2-Fusion(Ours)</td> 
   <td class="xl71">　</td> 
   <td class="xl65">61.33</td> 
   <td class="xl65">49.85</td> 
   <td class="xl65">49.12</td> 
   <td class="xl65">71.27</td> 
   <td class="xl65">61.24</td> 
   <td class="xl65">57.03</td> 
  </tr> <!--EndFragment--> 
 </tbody>
</table>
</div>



## Citation 

If you find this project useful in your research, please consider cite:


```
@ARTICLE{9991894,
  author={Wang, Li and Zhang, Xinyu and Li, Jun and Xv, Baowei and Fu, Rong and Chen, Haifeng and Yang, Lei and Jin, Dafeng and Zhao, Lijun},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={Multi-Modal and Multi-Scale Fusion 3D Object Detection of 4D Radar and LiDAR for Autonomous Driving}, 
  year={2023},
  volume={72},
  number={5},
  pages={5628-5641},
  doi={10.1109/TVT.2022.3230265}}
```



