import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
from pathlib import Path
from dotenv import load_dotenv

#---###---#

def get_credentials_file_path():
    credentials_file_script_path = Path(__file__).resolve()
    credentials_file_project_folder = credentials_file_script_path.parent.parent
    credentials_file_path = credentials_file_project_folder / "configs_scripts" / "auth_google_drive_config.json"

    try:
        with open(credentials_file_path, "r", encoding="utf-8") as file:
            content = file.read()
            
            if content:
                return credentials_file_path
            
    except FileNotFoundError as error:
        print(f"[ERROR] The file 'auth_google_drive_config.json' not found in: {credentials_file_project_folder} | {error}")
        
#---###---#

def get_general_infos_env_file_path():
    general_infos_env_path = Path(__file__).resolve()
    general_infos_env_project_folder = general_infos_env_path.parent.parent
    general_infos_env_file_path = general_infos_env_project_folder / "configs_scripts" / "general_infos.env"

    try:
        with open(general_infos_env_file_path, "r", encoding="utf-8") as file:
            content = file.read()
            
            if content:
                return general_infos_env_file_path
            
    except FileNotFoundError as error:
        print(f"[ERROR] The file 'general_infos.env' not found in: {general_infos_env_project_folder}| {error}")
    
#---###---#

def auth_drive():
    credentials_file_collected = get_credentials_file_path()
    
    load_dotenv(get_general_infos_env_file_path())
    scopes_collected = [f"{os.getenv("SCOPES")}"]
    auth_drive_creds = service_account.Credentials.from_service_account_file(credentials_file_collected,
                                                                             scopes=scopes_collected)
    auth_drive_service = build("drive",
                               "v3",
                               credentials=auth_drive_creds)
    return auth_drive_service