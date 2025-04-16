"""
setup.py \n
Enable installing through pip (preferably in editable mode)
"""

from setuptools import find_packages, setup

setup(
    name="MultiRefLfD-TPGP",
    version=open("VERSION").read().strip(),
    author="Ashrith Sagar",
    description="Learning Multi-Reference Frame Skills from Demonstration with Task-Parameterized Gaussian Processes (TPGP)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["lfd", "lfd.*"]),
    package_dir={"": "."},
    include_package_data=True,
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.9",
)
