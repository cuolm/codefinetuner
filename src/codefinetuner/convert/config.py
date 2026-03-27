import logging
import importlib.metadata
import httpx
from dataclasses import dataclass, field, fields
from pathlib import Path
from omegaconf import OmegaConf 


logger = logging.getLogger(__name__)


@dataclass
class Config:
    workspace_path: Path | None = None 
    convert_hf_to_gguf_local_path: Path = field(init=False)
    lora_model_path: Path = field(init=False)
    lora_model_gguf_path: Path = field(init=False)

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "Config":
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")
        
        logger.info(f"Loading configuration from {yaml_path}")
        config_dict = OmegaConf.structured(cls)
        try:
            yaml_file_node = OmegaConf.load(yaml_path)
        except Exception as e:
            raise ValueError(f"Failed to load YAML config {yaml_path}") from e

        yaml_file_dict = OmegaConf.to_container(yaml_file_node, resolve=True)
        yaml_convert_dict = yaml_file_dict.get("convert", {})

        yaml_convert_valid_dict = {}
        # Filter YAML fields to include only those defined in the Config dataclass.
        # This prevents OmegaConf from raising an AttributeError when encountering 
        # global YAML anchors or keys not present in the current Config dataclass. 
        for field in fields(cls):
            if field.name in yaml_convert_dict:
                yaml_convert_valid_dict[field.name] = yaml_convert_dict[field.name]
        logger.debug(f"Filtered YAML configuration: {yaml_convert_valid_dict}")

        merged_config_dict = OmegaConf.merge(config_dict, yaml_convert_valid_dict)
        return OmegaConf.to_object(merged_config_dict)
    
    def __post_init__(self) -> None:
        self._setup_paths()
        self._ensure_output_paths_exist()
        self._sync_converter_script_version()

    def _setup_paths(self) -> None:
        if self.workspace_path is None:
            self.workspace_path = Path.cwd()
            
        pkg_root = self.workspace_path / "src" / "codefinetuner" / "convert"
        self.convert_hf_to_gguf_local_path = pkg_root / "convert_hf_to_gguf.py"
        self.lora_model_path = self.workspace_path / "outputs" / "finetune" / "results" / "lora_model"
        self.lora_model_gguf_path = self.workspace_path / "outputs" / "convert" / "results" / "lora_model.gguf"

    def _ensure_output_paths_exist(self) -> None:
        parent_dir = self.lora_model_gguf_path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {parent_dir}")

    def _sync_converter_script_version(self) -> None:
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

        version_marker_path = self.convert_hf_to_gguf_local_path.with_suffix(".version")
        
        # check if we can skip the download
        if self.convert_hf_to_gguf_local_path.exists() and version_marker_path.exists():
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
                self.convert_hf_to_gguf_local_path.write_text(response.text, encoding="utf-8")
                version_marker_path.write_text(current_gguf_version)  # save the version marker to prevent re-downloading next time
                
        except Exception as e:
            if self.convert_hf_to_gguf_local_path.exists():
                logger.error(f"Failed to update convert_hf_to_gguf.py scprit but local copy exists. Proceeding. Error: {e}")
            else:
                raise RuntimeError(f"Failed to download convert_hf_to_gguf.py script and no local copy found") from e