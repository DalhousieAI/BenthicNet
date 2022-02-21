|pre-commit| |black|

BenthicNet
==========

BenthicNet is a dataset containing underwater photographs of benthic habitats on the seafloor.

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: black

Installing cartopy dependencies::

    sudo apt-get install proj-bin

    sudo apt-get install libproj-dev proj-data proj-bin libgeos-dev
    pip uninstall -y shapely
    pip install --no-binary shapely shapely

    conda create --name benthicnet -y
    conda activate benthicnet
    conda install -c conda-forge cartopy -y
    pip install -e .[plots]
