from setuptools import setup, find_packages

setup(
    name='latencylib',
    version='1.0.0',
    url='https://github.com:Flashgap/latency_lib',
    description='Latency helpers for Fruitz',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)
