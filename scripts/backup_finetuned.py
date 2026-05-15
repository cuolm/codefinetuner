import subprocess
from datetime import datetime
from pathlib import Path


def backup_folders(folders: list[Path], project_root_path: Path):
    backup_path = project_root_path / "backup"
    backup_path.mkdir(exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = backup_path / f"backup_{now_str}.tar.gz"

    existing_folders = []
    for folder in folders:
        if folder.exists():
            existing_folders.append(folder)
        else:
            print(f"Skipping {folder} — does not exist.")

    if not existing_folders:
        print("No folders to back up.")
        return

    print(f"Creating archive {archive_path}...")
    cmd = ["tar", "-czvf", str(archive_path), *[str(f) for f in existing_folders]]
    subprocess.run(cmd, check=True)
    print(f"Done. Created {archive_path}")


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