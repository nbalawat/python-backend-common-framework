"""
Setup file for Google Dataflow pipeline deployment.
This file is used by Apache Beam to package dependencies for Dataflow workers.
"""

from setuptools import setup, find_packages

# Read requirements from pyproject.toml or define them here
REQUIRED_PACKAGES = [
    "apache-beam[gcp]>=2.50.0",
    "google-cloud-storage>=2.10.0",
    "google-cloud-bigtable>=2.20.0",
    "google-cloud-bigquery>=3.11.0",
    "google-cloud-pubsub>=2.18.0",
    "google-cloud-monitoring>=2.15.0",
    "pydantic>=2.0.0",
    "avro>=1.11.0",
    "fastavro>=1.8.0",
]

setup(
    name="dataflow-pipelines-example",
    version="0.1.0",
    description="Google Dataflow pipeline examples for batch and streaming processing",
    author="Python Commons Team",
    author_email="team@pythoncommons.org",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.9",
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)