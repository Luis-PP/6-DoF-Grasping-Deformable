<p align="center">
  <h1 align="center">Real-Time 6-DoF Grasping of Deformable Poultry Legs in Cluttered Bins Using
Deep Learning and Geometric Feature Extraction</h1>
  <p align="center">
    <strong>Rekha Raja</strong>
    ·
    <strong>Luis Angel Ponce Pacheco</strong>
    ·
    <strong>Akshay K. Burusa</strong>
    ·
    <strong>Gert Kootstra</strong>
    ·
    <strong>Eldert J. van Henten</strong>
  </p>
</p>

<h2 align="center">
  Paper: 
  <a href="https://" target="_blank">IEEE</a> | 
  <a href="https://" target="_blank">ArXiv</a>
</h2>

https://github.com/akshaykburusa/gradientnbv/assets/127020264/dfa1f2a9-f07c-4af0-84ef-7ea20a7cb61b

<h3 align="center">
<video controls>
  <source src="Short_Simple_x4.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</h3>


## Installation


### Clone the repository

```
git clone https://github.com/Luis-PP/6-DoF-Grasping-Deformable.git
```

### Python packages

Quick install:
```
cd 6-DoF_Grasping_of_Deformable_Poultry_Legs
conda env create -f environment.yml
conda activate chickenlegenv
```
Manual install:
```
conda create -n chickenlegenv python==3.8
conda activate chickenlegenv
conda install -c conda-forge numpy=1.22.3 scipy matplotlib-base=3.7.2 
conda install -c conda-forge pandas=1.5.3 scikit-learn=1.2.2
conda install pytorch=1.10.0 torchvision=0.11.0 torchaudio=0.10.0 cpuonly -c pytorch
conda install -c conda-forge opencv=4.5.2 pycocotools=2.0.7
conda install -c conda-forge hydra-core=1.3.2 omegaconf=2.3.0 yacs=0.1.8
conda install -c conda-forge tqdm=4.66.4
conda install -c conda-forge opencv=4.5.2 pycocotools=2.0.7
conda install -c conda-forge hydra-core=1.3.2 omegaconf=2.3.0 yacs=0.1.8
conda install -c conda-forge tqdm=4.66.4
conda install -c conda-forge vtk=9.0.1 pyvista=0.34.0 matplotlib=3.7.5
conda install -c conda-forge absl-py=2.1.0
conda install -c conda-forge importlib-metadata=8.0.0 importlib-resources=6.1.1
conda install -c conda-forge pillow=9.0.1
conda install -c conda-forge tabulate=0.9.0
pip install detectron2==0.6+cpu fvcore==0.1.5.post20221221
pip install protobuf==5.27.2 google-auth==2.31.0 requests==2.32.3
pip install tensorboard==2.14.0 tensorboard-data-server==0.7.2
```


### Add large file
Go to https://drive.google.com/file/d/1cjJ5UrVO298LzIcI5di2yT2fm8wcMwsq/view?usp=sharing, download the model_final.pth file and paste it to the project (i.e., /6-DOF-GRASPING-DEFORMABLE/model_final.pth).

## Execute

Method 1 - Curvature maximization:
```
python GPE_Curv_Max_FC.py
```

Method 2 - Convex hull:
```
python GPE_Curv_ConvexHull.py
```

Comparison of methods:
```
python GPE_CH__FC.py
```

## Citation
```bibtex
@inproceedings{burusa2024gradient,
  title={Real-Time 6-DoF Grasping of Deformable Poultry Legs in Cluttered Bins Using
Deep Learning and Geometric Feature Extraction},
  author={Raja, Rekha and Ponce Pacheco, Luis Angel and Burusa, Akshay K and Kootstra, Gert and van Henten, Eldert J},
  booktitle={},
  pages={,
  year={},
  organization={}
}
```