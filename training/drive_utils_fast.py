import os
import io
import threading
import concurrent.futures
from PIL import Image
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class FastDriveImageLoader:
    def __init__(self, credentials_path, folder_id):
        self.credentials_path = credentials_path
        self.folder_id = folder_id
        self.service = None
        self._file_cache = {}
        self._lock = threading.Lock()
        self._index_attempted = False
        
    def authenticate(self):
        # Authenticate using the credentials file
        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        self.service = build('drive', 'v3', credentials=creds)
        
    def get_drive_files(self):
        """Fetches files and filters for images in Python (Vani Logic)."""
        if not self.service:
            self.authenticate()
            
        page_token = None
        
        # Prevent infinite retries
        self._index_attempted = True
        
        print(f"Building Drive file index for folder: {self.folder_id}...")
        while True:
            try:
                # 1. Fetch EVERYTHING in the folder (No mimeType filter in query)
                results = self.service.files().list(
                    q=f"'{self.folder_id}' in parents",
                    fields='nextPageToken, files(id, name, mimeType)',
                    pageSize=1000,
                    pageToken=page_token,
                    # Shared Drive Support
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True
                ).execute()
                
                files = results.get('files', [])
                
                # 2. Filter in Python (The "Crucial Part" you mentioned)
                with self._lock:
                    for file in files:
                        # Check if it looks like an image
                        if file.get('mimeType', '').startswith('image/'):
                            self._file_cache[file['name']] = file['id']
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            except Exception as e:
                print(f"Error listing files: {e}")
                break
        
        print(f"Index built: Found {len(self._file_cache)} images in Drive.")
        return self._file_cache
    
    def load_image_batch(self, filenames, max_workers=10):
        """Load multiple images concurrently from Drive."""
        if not self.service:
            self.authenticate()
            
        # FIX: Check flag, not just cache size, to prevent infinite loops if folder is empty
        if not self._file_cache and not self._index_attempted:
            self.get_drive_files()
        
        def load_single_image(filename):
            clean_name = os.path.basename(filename)
            
            try:
                if clean_name in self._file_cache:
                    file_id = self._file_cache[clean_name]
                    request = self.service.files().get_media(fileId=file_id)
                    file_data = request.execute()
                    
                    img = Image.open(io.BytesIO(file_data))
                    return filename, img
                else:
                    return filename, None
            except Exception as e:
                # print(f"Error loading {clean_name}: {e}")
                return filename, None
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_filename = {executor.submit(load_single_image, fn): fn for fn in filenames}
            for future in concurrent.futures.as_completed(future_to_filename):
                fname, image = future.result()
                results[fname] = image
        
        return results

# Global instance
drive_loader = None

def init_fast_drive_loader(credentials_path, folder_id):
    if not credentials_path or not folder_id:
        raise ValueError("credentials_path and folder_id are required.")
        
    global drive_loader
    drive_loader = FastDriveImageLoader(credentials_path, folder_id)
    return drive_loader

def get_images_batch_from_drive(image_links, batch_size=50):
    global drive_loader
    if not drive_loader:
        raise ValueError("Drive loader not initialized.")
    
    if isinstance(image_links, str):
        image_links = [image_links]
    
    all_images = {}
    for i in range(0, len(image_links), batch_size):
        batch = image_links[i:i+batch_size]
        batch_results = drive_loader.load_image_batch(batch)
        all_images.update(batch_results)
    
    return all_images