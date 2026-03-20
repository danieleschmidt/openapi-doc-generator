from setuptools import setup, find_packages

setup(
    name="openapi-gen",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["PyYAML>=6.0"],
    entry_points={"console_scripts": ["openapi-gen=openapi_gen.cli:main"]},
)
