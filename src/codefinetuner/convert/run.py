import subprocess
import sys
import logging
from typing import Any

from .config import Config


logger = logging.getLogger(__name__)


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
