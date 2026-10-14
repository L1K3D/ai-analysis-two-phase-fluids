import io
import numpy as np
import os
import cv2

from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
from dotenv import load_dotenv

from utils.utls_google_drive_auth import get_general_infos_env_file_path, auth_drive

#---###---#

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
        ).execute()
        
        list_images_files.extend(list_images_response.get('files', []))
        list_images_page_token = list_images_response.get('nextPageToken', None)
        if not list_images_page_token:
            break
        
    return list_images_files

#---###---#
    
def load_image_as_array(file_id_collected,
                        image_x_size_collected,
                        image_y_size_collected):
    load_image_as_array_service_collected = auth_drive()
    load_image_as_array_int_image_x_size = int(image_x_size_collected)
    load_image_as_array_int_image_y_size = int(image_y_size_collected)
    load_image_as_array_target_size_collected = (load_image_as_array_int_image_x_size, load_image_as_array_int_image_y_size)
    
    load_image_as_array_request = load_image_as_array_service_collected.files().get_media(fileId=file_id_collected)
    load_image_as_array_fh = io.BytesIO()
    load_image_as_array_downloader = MediaIoBaseDownload(load_image_as_array_fh, load_image_as_array_request)
    
    load_image_as_array_done = False
    while not load_image_as_array_done:
        load_image_as_array_status, load_image_as_array_done = load_image_as_array_downloader.next_chunk()
        
    load_image_as_array_fh.seek(0)
    
    load_image_as_array_img = Image.open(load_image_as_array_fh)
    load_image_as_array_img_array = np.array(load_image_as_array_img)
    load_image_as_array_img_resized = cv2.resize(load_image_as_array_img_array, load_image_as_array_target_size_collected)
    load_image_as_array_img_normalized = load_image_as_array_img_resized.astype('float32') / 255.0
    
    return load_image_as_array_img_normalized

#---###---#

def build_dataset(limit_per_class_collected,
                  folder_name_collected):
    
    X = []
    Y = []
    
    if (folder_name_collected == "expulsion") or (folder_name_collected == "exp"):
        images_files = list_images(folder_name_collected=folder_name_collected)
        
        if limit_per_class_collected:
            images_files = images_files[:limit_per_class_collected]
        
        for img in images_files:
            matriz = load_image_as_array(img['id'], 224, 224)
            X.append(matriz)
            Y.append(0)
            
        return np.array(X), np.array(Y)
            
    elif (folder_name_collected == "incubation") or (folder_name_collected == "inc"):
        images_files = list_images(folder_name_collected=folder_name_collected)
        
        if limit_per_class_collected:
            images_files = images_files[:limit_per_class_collected]
        
        for img in images_files:
            matriz = load_image_as_array(img['id'], 224, 224)
            X.append(matriz)
            Y.append(1)
            
        return np.array(X), np.array(Y)
        
    else:
        print("[INFO] None of the options selected in parameter 'folder_name_collected' are available. Please, select one of the valid options: 'expulsion', 'exp', 'incubation' or 'inc'")