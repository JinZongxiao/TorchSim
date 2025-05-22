from setuptools import setup, find_packages

setup(
    name="torchsim",
    version="0.1.0",
    author="Jin Zongxiao",
    description="A molecular dynamics simulation framework powered by PyTorch",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JinZongxiao/torchsim",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.1",
        ],
    },
) 