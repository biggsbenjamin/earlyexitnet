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

### Python requirements - included in package
installed onnx-1.8.1 

onnxruntime-1.7.0 

## Train Network Example

`python -m earlyexitnet.cli -m b_lenet -bbe 2 -jte 3 -rn "run notes example" -t1 0.75 -entr 0.01`

## Getting visual representation of the onnx graph
Use [netron](ihttps://netron.app/) viewer

