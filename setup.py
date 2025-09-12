from setuptools import find_packages, setup
from typing import List

# Constant for the requirements file
REQUIREMENTS_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements() -> List[str]:
    """
    This function will return the list of requirements
    from the requirements.txt file.
    """
    with open(REQUIREMENTS_FILE_NAME) as file_obj:
        requirements = file_obj.readlines()
        # Use list comprehension to remove the newline characters
        requirements = [req.replace("\n", "") for req in requirements]

        # The '-e .' in requirements.txt is used to install the local package
        # in editable mode, but it's not a package dependency itself.
        # We need to remove it from the list.
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name="CropYieldPrediction",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=get_requirements(), # Dynamically get requirements
)