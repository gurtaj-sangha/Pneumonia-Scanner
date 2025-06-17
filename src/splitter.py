import os
from pathlib import Path

ROOT = Path("data")
SPLITS = ["train", "val", "test"]

def split_pneumonia_images():
    """
    Split pneumonia images from viral_pneumonia folders into bacterial_pneumonia and viral_pneumonia
    based on filename patterns (bacteria vs virus in the name).
    """
    print("Starting")
    
    total_bacterial = 0
    total_viral = 0
    
    for split in SPLITS:
        
        
        source_folder = ROOT / split / "viral_pneumonia"
        
        if not source_folder.exists():
            print(f"  Skipping {split} - viral_pneumonia folder not found")
            continue
        
        
        bacterial_folder = ROOT / split / "bacterial_pneumonia"
        viral_folder = ROOT / split / "viral_pneumonia_new"
        
        bacterial_folder.mkdir(parents=True, exist_ok=True)
        viral_folder.mkdir(parents=True, exist_ok=True)
        
        bacterial_count = 0
        viral_count = 0
        unknown_count = 0
        
        
        for file_path in source_folder.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                filename = file_path.name.lower()
                
                if "_bacteria_" in filename:
                    
                    dest_path = bacterial_folder / file_path.name
                    file_path.rename(dest_path)
                    bacterial_count += 1
                    
                elif "_virus_" in filename:
                   
                    dest_path = viral_folder / file_path.name
                    file_path.rename(dest_path)
                    viral_count += 1
                    
                else:
                    print(f"  Warning: Unknown pattern in file {file_path.name}")
                    unknown_count += 1
        
        print(f"  Bacterial pneumonia images: {bacterial_count}")
        print(f"  Viral pneumonia images: {viral_count}")
        if unknown_count > 0:
            print(f"  Unknown pattern files: {unknown_count}")
        
        total_bacterial += bacterial_count
        total_viral += viral_count
        
       
        try:
            if source_folder.exists() and not any(source_folder.iterdir()):
                source_folder.rmdir()
                print(f"  Removed empty folder: {source_folder}")
        except OSError:
            print(f"  Could not remove folder {source_folder} (not empty)")
        
        
        if viral_folder.exists():
            final_viral_folder = ROOT / split / "viral_pneumonia"
            if not final_viral_folder.exists():
                viral_folder.rename(final_viral_folder)
                print(f"  Renamed viral_pneumonia_new to viral_pneumonia")
    
    print("Complete!")

if __name__ == "__main__":
    split_pneumonia_images()