<p align="center">
  
  <h3 align="center"><strong>BelHouse3D: A Benchmark Dataset for Assessing Occlusion Robustness in 3D Point Cloud Semantic Segmentation</strong></h3>

  <p align="center">
      <a href="https://scholar.google.com/citations?user=Opq90WAAAAAJ" target='_blank'>Umamaheswaran Raman Kumar</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com/citations?user=nTVNKgwAAAAJ" target='_blank'>Abdur Fayjie</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
      <a href="" target='_blank'>Jurgen Hannaert</a><sup>2</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com/citations?user=zwa-3rYAAAAJ" target='_blank'>Patrick Vandewalle</a><sup>1</sup>
      <br>
  <sup>1</sup>KU Leuven&nbsp;&nbsp;&nbsp;
  <sup>2</sup>3Frog
  </p>

  <img src="docs/figs/teaser.png" align="center" width="100%">

  <details>
    <summary>
    <font size="+1">Abstract</font>
    </summary>
    Large-scale 2D datasets have been instrumental in advancing machine learning; however, progress in 3D vision tasks has been relatively slow. This disparity is largely due to the limited availability of 3D benchmarking datasets. In particular, creating real-world point cloud datasets for indoor scene semantic segmentation presents considerable challenges, including data collection within confined spaces and the costly, often inaccurate process of per-point labeling to generate ground truths. While synthetic datasets address some of these challenges, they often fail to replicate real-world conditions, particularly the occlusions that occur in point clouds collected from real environments. Existing 3D benchmarking datasets typically evaluate deep learning models under the assumption that training and test data are independently and identically distributed (IID), which affects the models' usability for real-world point cloud segmentation. To address these challenges, we introduce the BelHouse3D dataset, a new synthetic point cloud dataset designed for 3D indoor scene semantic segmentation. This dataset is constructed using real-world references from 32 houses in Belgium, ensuring that the synthetic data closely aligns with real-world conditions. Additionally, we include a test set with data occlusion to simulate out-of-distribution (OOD) scenarios, reflecting the occlusions commonly encountered in real-world point clouds. We evaluate popular point-based semantic segmentation methods using our OOD setting and present a benchmark. We believe that BelHouse3D and its OOD setting will advance research in 3D point cloud semantic segmentation for indoor scenes, providing valuable insights for the development of more generalizable models. 
  </details>

</p>


## Outline
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Benchmark](#benchmark)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


## Installation



## Data Preparation
Please download [BelHouse3D dataset](https://doi.org/10.48804/ZS8D6K) and organize them as following:

```
code_root
└── data/
   └── belhouse3d/
      ├── raw/
      |  ├── belhouse3d_classnames.txt
      |  |
      |  ├── IID-nonoccluded/
      |  |  ├── House1/
      |  |  ├── :
      |  |  └── House32/
      |  | 
      |  └── OOD-occluded/
      |     ├── House27/
      |     ├── :
      |     └── House32/
      |
      └── processed/
         └── fsl/
            └── test/
               ├── IID-nonoccluded/
               |  ├── meta/
               |  └── test/
               |
               └── OOD-occluded/
                  ├── meta/
                  └── test/
```
### Process data for fully-supervised semantic segmentation
```
sh process_data_semseg.sh
```
New folders created after running the script
```
code_root
└── data/
   └── belhouse3d/
      └── processed/
         └── semseg/
            └── IID-nonoccluded/
            |  ├── meta/
            |  ├── train/
            |  ├── val/
            |  └── test/
            |
            └── OOD-occluded/
               ├── meta/
               └── test/
```

### Process data for few-shot semantic segmentation


## Benchmark

### Fully Supervised 3D Segmentation

### Few-Shot 3D Segmentation

## License

This project is released under the [MIT license](./LICENSE).


## Citation
If you find this work helpful, please consider citing the paper:

```bibtex
@misc{kumar2024belhouse3dbenchmarkdatasetassessing,
      title={BelHouse3D: A Benchmark Dataset for Assessing Occlusion Robustness in 3D Point Cloud Semantic Segmentation}, 
      author={Umamaheswaran Raman Kumar and Abdur Razzaq Fayjie and Jurgen Hannaert and Patrick Vandewalle},
      year={2024},
      eprint={2411.13251},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13251}, 
}
```

If you have utilized the dataset, please consider citing the dataset along with the paper:

```bibtex
@data{ZS8D6K_2024,
author = {Raman Kumar, Umamaheswaran and Hannaert, Jurgen and Vandewalle, Patrick},
publisher = {KU Leuven RDR},
title = {{BelHouse3D: A Dataset for 3D Indoor Scene Point Clouds}},
year = {2024},
version = {V1},
doi = {10.48804/ZS8D6K},
url = {https://doi.org/10.48804/ZS8D6K}
}
```


## Acknowledgement
We thank ... for sharing their source code.
