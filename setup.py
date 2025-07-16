from setuptools import setup, find_packages

setup(
    name='pfns4mvbo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        #'numpy',
        #'torch',
        #'mcbo'
    ],
    package_data={
        "pfns4mvbo": ["trained_models/*.pth"],
    },
    author='Timothy Shinners',
    description='PFNs For Mixed-Variable Bayesian allows for the use of PFNs as a surrogate function in Bayesian optimization methods',
    url='https://github.com/TimShinners/PFNs4MVBO/',  # optional
)



