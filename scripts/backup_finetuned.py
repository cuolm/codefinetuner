import shutil
from datetime import datetime
from pathlib import Path


def backup_folders(folders: list[Path], project_root_path: Path):
    backup_path = project_root_path / "backup"
    backup_path.mkdir(exist_ok=True)
    
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = backup_path / f"backup_{now_str}"
    backup_folder.mkdir(exist_ok=True)
    
    for folder in folders:
        if folder.exists():
            renamed_folder = f"{folder.name}_{now_str}"
            dest_folder = backup_folder / renamed_folder
            shutil.copytree(folder, dest_folder)
            print(f"Copied and renamed {folder} to {dest_folder}")
        else:
            print(f"Folder {folder} does not exist. Skipping.")


def main():
    project_root_path = Path(__file__).resolve().parent.parent
    folders_to_backup = [
        project_root_path / "config", 
        project_root_path / "src", 
        project_root_path / "outputs"
    ]
    backup_folders(folders_to_backup, project_root_path)


if __name__ == "__main__":
    main()
