from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-disaster-response",
    version="2.0.0",
    author="Modernized Implementation",
    author_email="",
    description="Modern multimodal deep learning for disaster response using PyTorch and Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/multimodal-disaster-response",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "disaster-train=train:main",
            "disaster-evaluate=evaluate:main",
            "disaster-predict=predict:main",
        ],
    },
)