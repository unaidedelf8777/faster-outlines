# Copyright 2024 Nathan Hoos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tomlkit
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_dependencies():
    pyproject_path = os.path.join(CURRENT_DIR, 'pyproject.toml')

    with open(pyproject_path, "r") as file:
        data = tomlkit.parse(file.read())
        dependencies = data["tool"]["poetry"]["dependencies"]
        return dependencies


metadata = {
    'name': "faster_outlines",
    'version': "09.18.2024",
    'description': "Faster backend for the `Outlines` library.",
    'long_description': "", 
    'authors': ["unaidedelf8777"],
    'author_email': "thwackyy.y@gmail.com",
    'license': "Apache 2.0",
    'classifiers': [
        "Programming Language :: Rust",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    'url': "https://github.com/unaidedelf8777/faster-outlines/",
}

rust_extensions = [
    RustExtension(
        "faster_outlines.fsm.fsm_utils",
        f"{CURRENT_DIR}/Cargo.toml",
        binding=Binding.PyO3,
        features=["default"],
        args=["--profile=release"]
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
