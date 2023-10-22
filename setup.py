from setuptools import setup, find_packages

setup(
  name = 'magvit2-pytorch',
  packages = find_packages(),
  version = '0.0.23',
  license='MIT',
  description = 'MagViT2 - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/magvit2-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformer',
    'attention mechanisms',
    'generative video model'
  ],
  install_requires=[
    'accelerate',
    'beartype',
    'einops>=0.7.0',
    'kornia',
    'vector-quantize-pytorch>=1.9.18',
    'torch',
    'torchvision'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
