# MNIST_RNET

## Installation
Install package
```bash
pip install git+ssh://git@github.com/phvan2312/MNIST_RNET.git
```

## Usage
The model uses CPU by default. No support for runtime changes for now. To run inference
```python
from inference import predict
predict('path_to_image')
```
