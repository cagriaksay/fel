"""
Setup script for FEL-CA
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fel-ca",
    version="1.0.0",
    author="Cagri Aksay",
    author_email="cagri@aksay.co",
    description="Deterministic Flux-Equality Cellular Automaton for Coherent Wave Dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cagriaksay/fel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
        "napari[all]>=0.4.18",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)

