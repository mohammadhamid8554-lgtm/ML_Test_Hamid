import  os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "ML_Project"
list_of_files = [

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/Components/__init__.py",
    f"src/{project_name}/Components/Data_Ingestion.py",
    f"src/{project_name}/Components/Data_Tranformation.py",
    f"src/{project_name}/Components/Model_Trainer.py",
    f"src/{project_name}/Components/Model_Monitering.py",
    f"src/{project_name}/Pipelines/__init__.py",
    f"src/{project_name}/Pipelines/Training_Pipeline.py",
    f"src/{project_name}/Pipelines/Prediction_Pipeline.py",
    f"src/{project_name}/Exception_Handling.py",
    f"src/{project_name}/Logger.py",
    f"src/{project_name}/Utils.py",
    "app.py",
    "Docker_File",
    "requirements.txt",
    "setup.py",
    "main.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Create Directory: {filedir} for the file {filename}")


    if (not os.path.exists(filename)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists!!")