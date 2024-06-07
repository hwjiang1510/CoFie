<div align="center">

# CoFie: Learning Compact Neural Surface Representations with Coordinate Fields

<p align="center">
    <a href="https://hwjiang1510.github.io/">Hanwen Jiang</a>,
    <a href="https://yanghtr.github.io/">Haitao Yang</a>,
    <a href="https://geopavlakos.github.io/">Georgios Pavlakos</a>,
    <a href="https://www.cs.utexas.edu/~huangqx/">Qixing Huang</a>
</p>

</div>

--------------------------------------------------------------------------------

<div align="center">
    <a href="https://hwjiang1510.github.io/CoFie/"><strong>Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2406.03417"><strong>Paper</strong></a>
</div>

<br>

## Installation
Please follow [DeepSDF](https://github.com/facebookresearch/DeepSDF) for environment installation.

## Pre-process
- Please follow [DeepSDF](https://github.com/facebookresearch/DeepSDF) for preprocess the shapes and get SDF samples (both SdfSamples and SurfaceSamples).
- Use ```python process_data.py``` to assign each SDF point sample to a local shape.
- Use ```python process_data_normal.py``` to pre-compute coordinate field initialization for each local shape.

## Training and Testing
- Use ```./train.sh``` for training
- Use ```./reconstruct.sh``` for testing. Use ```./reconstruct_batch.py``` for fast validation.

## Pre-trained Model
The trained MLP can be found [here](https://utexas.box.com/s/yw8fitj23n92jj3z9xxcztzzr7krkvay)


## Acknowledgement
This repo is developed from [DeepSDF](https://github.com/facebookresearch/DeepSDF) and [DeepLS](https://github.com/Kamysek/DeepLocalShapes)
