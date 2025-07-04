import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HidKim_K2IE",
    version="0.0.1",
    author="Hideaki Kim",
    author_email="hideaki.kin@ntt.com",
    license="SOFTWARE LICENSE AGREEMENT FOR EVALUATION",
    description="Kernel method-based kernel intensity estimator implemented in Tensorflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HidKim/K2IE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
