from setuptools import setup, find_packages
import os
import re

# Read the version from the __init__.py file
with open(os.path.join("microgpt", "__init__.py"), "r") as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    version = version_match.group(1) if version_match else "0.1.0"

# Read README for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="microgpt",
    version=version,
    description="A lightweight, modular framework for building composable AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MicroGPT Team",
    author_email="example@example.com",
    url="https://github.com/username/microgpt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "requests>=2.25.0",
        "python-dotenv>=0.15.0",
        "beautifulsoup4>=4.9.0",
        "tiktoken>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
    },
    keywords=["ai", "agents", "llm", "openai", "gpt", "framework"],
)