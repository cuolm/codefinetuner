import subprocess
import sys
import logging
import importlib.metadata
import httpx
from typing import Any

from .config import Config


logger = logging.getLogger(__name__)


def _ensure_output_paths_exist(config) -> None:
    parent_dir = config.lora_model_gguf_path.parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {parent_dir}")


def _sync_converter_script_version(config) -> None:
    """
    Downloads the convert_hf_to_gguf.py script from llama.cpp github repo
    if it is missing or version mismatched with localy installed gguf package (in .venv).
    """
    try:
        current_gguf_version = importlib.metadata.version("gguf")
        version_tag = f"gguf-v{current_gguf_version}"
    except importlib.metadata.PackageNotFoundError:
        logger.warning("gguf package not found. Defaulting to master branch.")
        version_tag = "master"
        current_gguf_version = "master"

    version_marker_path = config.convert_hf_to_gguf_local_path.with_suffix(".version")
    
    # check if we can skip the download
    if config.convert_hf_to_gguf_local_path.exists() and version_marker_path.exists():
        last_version = version_marker_path.read_text().strip()
        if last_version == current_gguf_version:
            logger.debug(f"convert_hf_to_gguf.py script is up to date (version {current_gguf_version}).")
            return

    # proceed with download
    github_url = f"https://raw.githubusercontent.com/ggerganov/llama.cpp/{version_tag}/convert_hf_to_gguf.py"
    logger.info(f"Syncing convert_hf_to_gguf.py script for version {version_tag}...")
    
    try:
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(github_url)
            if response.status_code == 404 and version_tag != "master":
                logger.warning(f"Version {version_tag} not found. Falling back to master.")
                response = client.get("https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py")
            
            response.raise_for_status()
            config.convert_hf_to_gguf_local_path.write_text(response.text, encoding="utf-8")
            version_marker_path.write_text(current_gguf_version)  # save the version marker to prevent re-downloading next time
            
    except Exception as e:
        if config.convert_hf_to_gguf_local_path.exists():
            logger.error(f"Failed to update convert_hf_to_gguf.py scprit but local copy exists. Proceeding. Error: {e}")
        else:
            raise RuntimeError(f"Failed to download convert_hf_to_gguf.py script and no local copy found") from e
        

def _log_subprocess_output(process: Any, prefix: str = "llama.cpp") -> None:
    """
    Captures subprocess output, handles multi-line blocks,
    and re-logs them with correct severity levels.
    """
    log_levels = ("INFO:", "WARNING:", "ERROR:", "DEBUG:")
    message_buffer = []

    def flush_buffer() -> None:
        if message_buffer:
            msg = "".join(message_buffer).strip()
            if msg.startswith("ERROR:"):
                logger.error(f"[{prefix}] {msg}")
            elif msg.startswith("WARNING:"):
                logger.warning(f"[{prefix}] {msg}")
            else:
                logger.info(f"[{prefix}] {msg}")
            message_buffer.clear()

    for line in process.stdout:
        # check if the current line starts a new log entry
        if any(line.startswith(level) for level in log_levels):
            flush_buffer()
        
        message_buffer.append(line)

    flush_buffer()


def _convert_to_gguf(config: Config, precision: str = "auto"):
    cmd = [
        sys.executable, str(config.convert_hf_to_gguf_local_path),
        str(config.lora_model_path), 
        "--outfile", str(config.lora_model_gguf_path),
        "--outtype", precision
    ]
    
    logger.info(f"Executing conversion: {config.lora_model_path} -> {config.lora_model_gguf_path}")

    # initialize subprocess 
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,  # catch output in a private memory buffer (pipe) instead of letting it hit the console
        stderr=subprocess.STDOUT,  # merge error messages into that same private memory buffer (pipe)
        text=True,  # return streams as text (str) instead of bytes
        bufsize=1  # send text line-by-line so we see updates immediately
    )

    _log_subprocess_output(process=process, prefix="llama.cpp")

    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"Conversion failed with exit code {process.returncode}")


def run(config: Config) -> None:
    _ensure_output_paths_exist(config)
    _sync_converter_script_version(config)
    _convert_to_gguf(config)


def main() -> None:
    try:
        convert_config = Config()
        run(convert_config)
    except Exception:
        logger.exception("Model conversion to GGUF failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
