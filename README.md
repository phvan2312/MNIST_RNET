# MNIST_RNET

## Installation
Install package
```bash
pip install git+https://github.com/phvan2312/MNIST_RNET.git
```

Set up model path
```bash
export MNIST_MODEL_PATH='./MNIST_saved_model.t7'
```

## Usage
The model uses CPU by default. No support for runtime changes for now. To run inference
```python
from inference import predict
predict('path_to_image')
```
