import os
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

metadata = {
    'name': "faster_outlines",
    'version': "2024.11.10",
    'description': "Faster, lazy backend for the `Outlines` library.",
    'long_description': "", 
    'authors': ["Nathan Hoos"],
    'author_email': "thwackyy.y@gmail.com",
    'license': "Apache 2.0",
    'classifiers': [
        "Programming Language :: Rust",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    'url': "https://github.com/unaidedelf8777/faster-outlines/",
}

rust_extensions = [
    RustExtension(
        "faster_outlines.lib",
        f"{CURRENT_DIR}/rust/faster_outlines_rs/Cargo.toml",
        binding=Binding.PyO3,
        args=["--profile=release", "--features=python_bindings"]
    ),
]

setup(
    name=metadata['name'],
    version=metadata['version'],
    author=metadata['authors'][0].split(" <")[0],
    author_email=metadata['author_email'],
    description=metadata['description'],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url=metadata['url'],
    classifiers=metadata['classifiers'],
    license=metadata['license'],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.9',
    setup_requires=["setuptools>=69.5.1", "wheel", "setuptools-rust>=1.9.0", "tomlkit>=0.12.5"],
    rust_extensions=rust_extensions,
)
