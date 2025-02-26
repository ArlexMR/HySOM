from setuptools import setup, find_packages

setup(
    name = "HystSOM",
    version= "0.1.0",

    packages=find_packages(),
    #package_dir={'':"HystSOM"},
    include_package_data=True,
    package_data={"": ["data/*"]},
    install_requires=[
    "numpy >=1.26"]
)