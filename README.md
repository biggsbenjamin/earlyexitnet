# Early-exit Network(s)

## Conda Environment
Requires conda/miniconda for package management.
To set up the environment: 

`conda env create --file env.yaml`

## Other python requirements
installed onnx-1.8.1 

onnxruntime-1.7.0 

protobuf-3.15.5

Can be installed from conda env with:

`pip install onnx onnxruntime`

## Getting visual representation of the onnx graph
Make sure to activate the main env - with graphviz and pydot support.

Run the following to generate the dot file of the onnx graph:

`python <path to onnx>/tools/net_drawer.py --input <path to .onnx> --output <net name>.dot --embed_docstring`

Run the following to convert the dot file to svg:

`dot -Tsvg <net name>.dot -o <net name>.svg`
