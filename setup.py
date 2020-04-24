from os.path import abspath, dirname, join
from setuptools import setup

here = abspath(dirname(__file__))

with open(join(here, 'README.md')) as f:
    readme = f.read()

with open(join(here, 'LICENSE')) as f:
    lic = f.read()

setup(name='adaptive_confidence_intervals',
      version='0.1',
      description='Confidence Intervals for Policy Evaluation in Adaptive Experiments',
      author='Vitor Hadad and Ruohan Zhan',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/gsbDBI/adaptive-confidence-intervals',
      py_modules=['adaptive_CI'],
      install_requires=[
          "flake8==3.7.8",
          "ipykernel>=5.1.2",
          "jupyterlab>=1.1.4",
          "matplotlib>=3.1.1",
          "numpy>=1.17.0",
          "pandas>=0.25.0",
          "scipy>=1.3.0",
          "seaborn==0.9.0",
          "scikit-learn",
      ],
      classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          # Uncomment once license it added
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3'
      ],
      license=lic)
