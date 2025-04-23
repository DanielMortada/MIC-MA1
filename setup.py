from setuptools import setup, find_packages

setup(
    name="mic",
    version="0.1.0",
    description="Musical Instrument Classification using Deep Learning",
    author="Daniel",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8.0",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "tqdm>=4.62.0",
        "pillow>=9.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
