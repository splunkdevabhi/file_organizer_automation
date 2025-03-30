#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="file_organizer_automation",
    version="0.1.0",
    author="Abhishek Shukla",
    author_email="splunkdevabhi@github.com",
    description="A tool for organizing files automatically",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/splunkdevabhi/file_organizer_automation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "click>=7.0",  # For command line interface
        "watchdog>=2.0.0",  # For file system monitoring
        "colorlog>=6.0.0",  # For colored logging output
    ],
    entry_points={
        "console_scripts": [
            "organize=file_organizer.core.organizer:main",
            "organize-setup=file_organizer.scripts.startup:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="file organization, automation, directory management",
    project_urls={
        "Bug Tracker": "https://github.com/splunkdevabhi/file_organizer_automation/issues",
        "Documentation": "https://github.com/splunkdevabhi/file_organizer_automation",
        "Source Code": "https://github.com/splunkdevabhi/file_organizer_automation",
    },
)

