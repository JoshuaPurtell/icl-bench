from setuptools import find_packages, setup

setup(
    name="icl-bench",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "apropos-ai>=0.3.0",
        "smallbench>=0.1.17",
    ],
    author="Josh Purtell",
    author_email="jmvpurtell@gmail.com",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JoshuaPurtell/icl-bench",
    license="MIT",
)