import io
import numpy as np
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
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

def list_images(folder_name_collected):
    load_dotenv(get_general_infos_env_file_path())
    service_collected = auth_drive()
    
    if (folder_name_collected == "expulsion") or (folder_name_collected == "exp"):
        EXPULSION_FOLDER_ID = os.getenv("EXPULSION_FOLDER_ID")
        list_images_query =  f"'{EXPULSION_FOLDER_ID}' in parents and trashed = false"
        
    elif (folder_name_collected == "incubation") or (folder_name_collected == "inc"):
        INCUBATION_FOLDER_ID = os.getenv("INCUBATION_FOLDER_ID")
        list_images_query =  f"'{INCUBATION_FOLDER_ID}' in parents and trashed = false"
        
    else:
        print("[INFO] None of the options selected in parameter 'folder_name_collected' are available. Please, select one of the valid options: 'expulsion', 'exp', 'incubation' or 'inc'")
        
    list_images_files = []
    list_images_page_token = None
    
    while True:
        list_images_response = service_collected.files().list(
            q=list_images_query,
            pageSize=1000,
            fields="nextPageToken, files (id, name)",
            pageToken=list_images_page_token
        ).execute
        
        list_images_files.extend(list_images_response.get('files', []))
        list_images_page_token = list_images_response.get('nextPageToken', None)
        if not list_images_page_token:
            break
        
        return list_images_files
    
def load_image_as_array(service_collected,
                        file_id_collected):
    load_image_as_array_request = service_collected.files().get_media(fileId=file_id_collected)
    load_image_as_array_fh = io.BytesIO()
    load_image_as_array_downloader = MediaIoBaseDownload(load_image_as_array_fh, load_image_as_array_request)
    
    load_image_as_array_done = False
    while not load_image_as_array_done:
        load_image_as_array_status, load_image_as_array_done = load_image_as_array_downloader.next_chunk()
        
    load_image_as_array_fh.seek(0)
    
    load_image_as_array_img = Image.open(load_image_as_array_fh)
    load_image_as_array_img_array = np.array(load_image_as_array_img)
    
    return load_image_as_array_img_array

