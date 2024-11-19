from setuptools import setup, find_packages

setup(
    name='latencylib',
    version='1.0.0',
    url='https://github.com:Flashgap/latency_lib',
    description='Latency helpers for Fruitz',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.11.1',
        'matplotlib >= 1.5.1',
        'pandas >= 0.19.2',
        'plotly >= 4.14.3',
        'joblib >= 0.9.4',
        'google-cloud-bigquery >= 3.27.0',
    ],
)
