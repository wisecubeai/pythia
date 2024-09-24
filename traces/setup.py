from setuptools import setup, find_packages

setup(
    name="pythia-traces",  # This is the package name on PyPI
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"traces": "traces"},
    install_requires=["openlit==1.22.0"],
    python_requires=">=3.10",
    description="The Pythia Traces package for event tracing",
    url="https://wisecube.atlassian.net/wiki/spaces/OR/pages/147128324/AI+Hallucination+API+Pythia",
    author="Wisecube.ai",
    author_email="info@wisecube.ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
