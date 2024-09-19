# Open3DFDUNet: Open-Source 3D Fully-Dense U-Net
## Setup

### Installation
```
git clone https://github.com/thu-bplab/Open3DFDUNet.git
cd Open3DFDUNet
```

### Environment
Install requirements for Open3DFDUNet first.
```
pip install -r requirements.txt
```

## Training
### Tips
- The recommended PyTorch version is `>=2.0`. Code is developed and tested under PyTorch `2.0.1`.
- If you encounter CUDA OOM issues, please try to reduce the `batch_size` in the training and inference configs.
- The details related to the original network are all in [Model Detail](models/FDUNet3D.py). Each processing step is explained with comments, which corresponds to Figure 1 of the original paper.

### Configuration
- Our sample training defaults to use 4 gpus with `fp32` precision.
- You may modify the configuration to fit your own environment.

### Run Training
- Please replace data related paths in the script file or `train.py` with your own paths and customize the training settings.
- An example training usage is as follows:
  ```
  # Example usage
  cd scripts
  bash train.sh
  ```

### Inference on Trained Models
- You need to modify the `$MODEL_PATH` in the testing script.
- An example inference usage is as follows:
  ```
  bash test.sh
  ```

## Acknowledgement

- We thank the authors of the [original paper](https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202301277) for their great work!
- This project is supported by Beijing Academy of Artificial Intelligence by providing the computing resources.
- This project is advised by Cheng Ma.

## Citation
```
@article{zheng2023deep,
  title={Deep learning enhanced volumetric photoacoustic imaging of vasculature in human},
  author={Zheng, Wenhan and Zhang, Huijuan and Huang, Chuqin and Shijo, Varun and Xu, Chenhan and Xu, Wenyao and Xia, Jun},
  journal={Advanced Science},
  volume={10},
  number={29},
  pages={2301277},
  year={2023},
  publisher={Wiley Online Library}
}
```
```
@misc{open3dfdunet,
  title = {Open3DFDUNet: Open-Source 3D Fully-Dense U-Net},
  author = {Di Kong and Yuwen Chen},
  year = {2024},
  howpublished = {\url{https://github.com/thu-bplab/Open3DFDUNet}},
}
```
