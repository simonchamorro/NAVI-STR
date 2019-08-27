from setuptools import find_packages, setup

setup(
    name='ocr',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['scikit-optimize', 'opencv-python', 'pandas', 'tqdm',
                      'tables', 'numpy', 'torch', 'torchvision',
                      'matplotlib', 'tensorflow', 'tensorboard',
                      'terminaltables', 'pillow', 'tqdm', 'lmdb',
                      'nltk', 'natsort', 'easydict']
)
