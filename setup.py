from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension


with open("README.md", mode="r", encoding="utf-8") as readme:
    long_description = readme.read()


setup(
    name='bm25',
    version="0.0.0.2",
    description='BM25 in Rust',
    long_description="No Description",
    long_description_content_type="text/markdown",
    url="https://github.com/hkjeon13/bm25-rs.git",
    author="Eddie",
    author_email="hkjeo13@gmail.com",
    zip_safe=False,
    license="MIT",
    rust_extensions=[RustExtension("bm25/bm25", binding=Binding.PyO3)],
    py_modules=["bm25"],

    python_requires=">=3",

    packages=find_packages("."),
    package_data={"": ["*.json"]},
    include_package_data=True,
    install_requires=[],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],


)
