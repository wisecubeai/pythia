import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="pythia",
    version="1.0.0",
    author="Wisecube.ai",
    author_email="info@wisecube.ai",
    description="AI Hallucination toll",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://wisecube.atlassian.net/wiki/spaces/OR/pages/147128324/AI+Hallucination+API+Pythia",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    package_dir={"pythia": "pythia"},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.10",
)
