from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirement(file_path: str) -> List[str]:

    '''
    This function will return the list of requirements
    '''
    requirements = []

    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements= [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            return requirements.remove(HYPEN_E_DOT) # type: ignore

    return requirements



setup(
    name= "ML_Project",
    version= "0.0.1",
    author= "Mohammed Hamid",
    author_email= "mohammadhamid8554@gmail.com",
    packages= find_packages(),
    install_requires=get_requirement("requirements.txt")
)