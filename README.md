# Early-exit Network(s)
Hopefully going to be a repository of EE models that I can work with in pytorch.
`BranchyNet.py` is based on the branchy-LeNet model from the [BranchyNet](https://github.com/kunglab/branchynet) repo.

## Conda Environment
Requires conda/miniconda for package management.
To set up the environment: 

`conda env create --file pt1_8_env.yaml`

## Other python requirements
installed onnx-1.8.1 

onnxruntime-1.7.0 

Can be installed from conda env with:

`pip install onnx onnxruntime`

## Getting visual representation of the onnx graph
Use [netron](ihttps://netron.app/) viewer

## Running onnx test script
Should work from environment `pt1_8_env.yaml`

Requires path to trained network for branchynet, not required for testnet/lenet.

Command:
`python test_onnx.py --model [brn, lenet] --trained_path path/to/trained/branchynet.pth --save_name onnx_save_name`

Example:
`python test_onnx.py --model brn --trained_path /home/localadmin/phd/earlyexitnet/outputs/pre_Trn_bb_2021-07-09_141616/pretrn-joint-8-2021-07-09_142311.pth --save_name test_out`
