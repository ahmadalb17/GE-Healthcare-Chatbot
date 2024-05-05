import os 
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

#######################################
# Define the folder for storing databse 
#######################################

# Get the value of the presist derectory 
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')
if PERSIST_DIRECTORY is None:
    raise("==> set the PERSIST_DIRECTORY enviroment variabe")


# Define the chroma settings 
CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)

