import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from omegaconf import OmegaConf, MISSING


logger = logging.getLogger(__name__)


@dataclass
class Config:
    model_name: str = MISSING  
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

    def _setup_paths(self) -> None:
        if self.workspace_path is None:
            self.workspace_path = Path.cwd()
            
        pkg_root = self.workspace_path / "src" / "codefinetuner" / "convert"
        self.convert_hf_to_gguf_local_path = pkg_root / "convert_hf_to_gguf.py"
        self.lora_model_path = self.workspace_path / "outputs" / "finetune" / "results" / "lora_model"

        model_short_name = self.model_name.split("/")[-1]
        model_name_gguf = f"{model_short_name}-lora-merged.gguf"
        self.lora_model_gguf_path = self.workspace_path / "outputs" / "convert" / "results" / model_name_gguf

