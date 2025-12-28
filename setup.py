from setuptools import setup, find_packages

setup(
    name="global-lorenz",
    version="0.1.0",
    description="Fit global Lorenz curves to World Bank Data",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.7",
)
