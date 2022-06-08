from setuptools import find_packages, setup

setup(
    name="np_uptake",
    version="0.1.0",
    author="Sarah Iaquinta",
    author_email="sarah.r.iaquinta@gmail.com",
    packages=find_packages(),
    url="https://github.com/SarahIaquinta/adaptative_uptake_of_circular_np",
    description="Package to create and validate surrogate model and to compute Sobol's sensitivity indices",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
