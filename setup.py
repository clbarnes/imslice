from pathlib import Path
from setuptools import setup, find_packages

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

setup(
    name="imslice",
    url="https://github.com/clbarnes/imslice",
    author="Chris L. Barnes",
    description="Slice a 3D volume with a 2D arbitrary plane",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["imslice*"]),
    install_requires=["numpy", "scipy"],
    python_requires=">=3.7, <4.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
