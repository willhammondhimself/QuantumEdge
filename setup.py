"""Setup script for QuantumEdge."""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantumedge",
    version="0.1.0",
    author="QuantumEdge Team",
    author_email="team@quantumedge.io",
    description="Quantum-inspired portfolio optimization with crisis-proof robustness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantumedge/quantumedge",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.4", "black>=23.12.1", "mypy>=1.8.0"],
    },
    entry_points={
        "console_scripts": [
            "quantumedge=quantumedge.cli:main",
        ],
    },
)