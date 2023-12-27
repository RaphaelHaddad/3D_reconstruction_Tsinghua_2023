from distutils.core import setup
import os
from setuptools import find_packages

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if "__pycache__" not in path and not filename.endswith(".txt") and not filename.endswith(".pth") and not filename.endswith(".JPG"):
                paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('./kp_imc23')

setup(
  name = 'kp_imc23',         # How you named your package folder (MyLib)
  packages=find_packages(exclude=['*.pth']),
  package_data={'': extra_files},
  version = '0.14',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Package for the Image Matching Challenge 2023',   # Give a short description about your library
  author = 'Kevin Pruvost \ Raphael El Haddad \ Borislav Pavlov',                   # Type in your name
  author_email = 'pruvostkevin0@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/RaphaelHaddad/3D_reconstruction_Tsinghua_2023',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['CV', 'IMC'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
        'kornia',
        'opencv-python',
        'einops',
        'Pillow',
        'pandas',
        'opencv-python-headless',
        'numpy',
        'matplotlib',
        'torchmetrics',
        'scikit-learn',
        'tensorflow',
        'torch',
        'transformers',
        'pycolmap',
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.9',
  ],
)