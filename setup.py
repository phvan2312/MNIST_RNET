from setuptools import setup

requirements = open('requirements.txt').read().splitlines()

setup(name='MNIST_RNET',
      description='Numbers + Yen sign classification .',
      version='0.1.0',

      packages=['mnist_rnet','mnist_rnet.models'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)
