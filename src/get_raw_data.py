import os
import zipfile
import tarfile
from pathlib import Path
from config import PATHS
import requests


def download_from_yandex_disk(public_url: str, save_path: str):
    """
    Скачивает файл с Яндекс.Диска
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        download_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={public_url}"
        response = requests.get(download_url)
        
        if response.status_code == 200:
            href = response.json().get('href')
            if href:
                file_response = requests.get(href, stream=True)
                file_response.raise_for_status()
                
                total_size = int(file_response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(save_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                percent = (downloaded / total_size) * 100
                                print(f"\rПрогресс: {percent:.1f}%", end='')
                
                print(f"\nФайл сохранён: {save_path}")
                return True
        return False
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def extract_archive(archive_path: str, extract_to: str):
    """
    Разархивирует файл
    """
    os.makedirs(extract_to, exist_ok=True)
    archive_path = Path(archive_path)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                
        elif archive_path.suffix in ['.tar', '.gz', '.tgz'] or '.tar.' in archive_path.name:
            if archive_path.suffix == '.gz' or archive_path.suffix == '.tgz':
                mode = 'r:gz'
            elif archive_path.suffix == '.bz2':
                mode = 'r:bz2'
            else:
                mode = 'r'
                
            with tarfile.open(archive_path, mode) as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            return False
            
        print(f"Данные разархивированы в: {extract_to}")
        return True
        
    except Exception as e:
        print(f"Ошибка разархивирования: {e}")
        return False


if __name__ == "__main__":
    url = "https://disk.yandex.ru/d/o4oQ4Yk7SpH1dw"
    archive_path = PATHS['raw_data']
    extract_path = PATHS['extracted_data']
    
    if download_from_yandex_disk(url, archive_path):
        extract_archive(archive_path, extract_path)