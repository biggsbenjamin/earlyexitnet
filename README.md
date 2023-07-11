# Early-exit Network(s)
Hopefully going to be a repository of EE models that I can work with in pytorch.
`BranchyNet.py` is based on the branchy-LeNet model from the [BranchyNet](https://github.com/kunglab/branchynet) repo.

## Python Setup
Recommeded conda/miniconda for package management.

1. Set up a python 3.9 environment and activate it:

```
conda create -n py39 python=3.9
conda activate py39
```

2. Upgrade to latest version of pip.

`python -m pip install --upgrade pip`

3. Install package from current directory (earlyexitnetwork):

`pip install .`

### Requirements
- torch 1.13.1 (for CUDA >=11.6)
- onnx 1.8.1 
- onnxruntime 1.7.0 

This version of ONNX in python is old so requires protobuf compiler to be installed.

For Ubuntu this can be done with:

`sudo apt install protobuf-compiler libprotoc-dev`

Then, re-run `pip install .`

> **Note** Issues with pip failing may be solved by `conda install [package]=[version]` specified in the `pyproject.toml`

### Troubleshooting

For other Distros you may need a more recent version.

Check the installed version using `protoc --version`

The protobuf version required >= 3.5 and can be [built from source](https://pypi.org/project/onnx/) if necessary.

cmake version required >= 3.1 and can be installed to conda using `conda install cmake`

## Train & Test Network Example

`python -m earlyexitnet.cli -m [model name] -bbe [backbone epochs] -jte [joint exit & backbone epochs] -rn "run notes example" -t1 0.75 -entr 0.01`

`python -m earlyexitnet.cli -m b_lenet_se -bbe 50 -jte 30 -rn "run notes example" -t1 0.75 -entr 0.01`

## Test Only Example

`python -m earlyexitnet.cli -m b_lenet -mp /path/to/saved/model.pth -rn "run notes example" -t1 0.75 -entr 0.01`

This sets the top1 (maximum softmax) threshold to 0.75 and the entropy threshold to 0.01.

## Convert Model to ONNX Example

`python -m earlyexitnet.cli -m b_lenet -mp /path/to/saved/model.pth -rn "run notes example" -go path/to/onnx/folder/`

## List of Models

TBD

## Getting visual representation of the onnx graph
Use [netron](ihttps://netron.app/) viewer

