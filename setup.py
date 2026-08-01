from setuptools import find_packages, setup
from typing import List

REQUIREMENTS_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements() -> List[str]:
    """Read and return dependencies from requirements.txt."""
    with open(REQUIREMENTS_FILE_NAME) as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="CropYieldPrediction",
    version="1.0.0",
    author="Niloy",
    description="AI-powered crop yield prediction with disease detection extension",
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires=">=3.10",
)