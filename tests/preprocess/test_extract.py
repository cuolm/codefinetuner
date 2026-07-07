import pathlib
import json
import textwrap
from pathlib import Path

import pytest


tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.preprocess.run import _ensure_output_paths_exist
from codefinetuner.preprocess.config import Config
from codefinetuner.preprocess.extract import (
    auto_create_split_paths,
    _log_split_paths,
    _extract_code_blocks_rec,
    get_code_blocks_from_paths,
    get_code_blocks_from_auto_split,
    _check_required_directories,
    _get_filtered_paths,
    get_code_blocks_from_manual_split
)


# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> Config:
    """Load a Config from the test YAML, redirecting outputs to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config.raw_data_path = None  # set to none, recalculate it in _setup_paths()
    test_config._setup_paths()  # regenerates paths relative to the new workspace_path
    return test_config


# --- auto_create_split_paths ---

def test_auto_create_split_paths_partitions_all_files(config):
    """auto split on a flat directory should account for all files."""
    config.raw_data_path.mkdir(parents=True, exist_ok=True)
    
    num_files = 10
    for i in range(num_files):
        tmp_file = config.raw_data_path / f"file_{i}.c"
        tmp_file.write_text(f"int f{i}(){{}}", encoding="utf-8")

    config.split_mode = "auto"
    config.train_ratio = 0.7
    config.eval_ratio = 0.2
    config.test_ratio = 0.1
    
    _ensure_output_paths_exist(config)

    train, eval, test = auto_create_split_paths(config)
    assert len(train) == 7
    assert len(eval) == 2
    assert len(test) == 1


# --- _log_split_paths ---

def test_log_split_paths(config):
    _ensure_output_paths_exist(config)
    
    train_paths = [Path("first_train_path"), Path("second_train_path")]
    eval_paths = [Path("first_eval_path"), Path("second_eval_path")]
    test_paths = [Path("first_test_path"), Path("second_test_path")]

    _log_split_paths(config, train_paths, eval_paths, test_paths)

    assert config.split_log_path.exists()

    with config.split_log_path.open("r", encoding="utf-8") as file:
        split_log = json.load(file)

    assert split_log["train"][0] == str(train_paths[0])
    assert split_log["train"][1] == str(train_paths[1])
    assert split_log["eval"][0] == str(eval_paths[0])
    assert split_log["eval"][1] == str(eval_paths[1])
    assert split_log["test"][0] == str(test_paths[0])
    assert split_log["test"][1] == str(test_paths[1])


# --- _extract_code_blocks_rec ---

def test_extract_code_blocks_rec_yields_correct_blocks(config):
    source_code_utf8 = textwrap.dedent("""
        int calculate_power(int base, int exp) {
            if (exp <= 0) {
                return 1;
            }
            return base * calculate_power(base, exp - 1);
        }
    """).encode('utf8')
    tree = config.tree_sitter_parser.parse(source_code_utf8)
    max_depth = 10
    blocks = list(_extract_code_blocks_rec(config,  tree.root_node, source_code_utf8, max_depth))
    expected_block = b'int calculate_power(int base, int exp) {\n    if (exp <= 0) {\n        return 1;\n    }\n    return base * calculate_power(base, exp - 1);\n}'
    assert len(blocks) == 1
    assert blocks[0][0] == expected_block
    assert blocks[0][1].type == "function_definition"
    assert blocks[0][1].start_point == (1, 0)
    assert blocks[0][1].end_point == (6, 1)


# --- get_code_blocks_from_paths ---

def test_get_code_blocks_from_paths_yields_correct_blocks(config):
    tmp_cfile_path = config.workspace_path / "test_file.c"
    tmp_cfile_text = textwrap.dedent("""
        int calculate_power(int base, int exp) {
            if (exp <= 0) {
                return 1;
            }
            return base * calculate_power(base, exp - 1);
        }
    """)
    tmp_cfile_path.write_text(tmp_cfile_text)
    tmp_file_paths = [tmp_cfile_path]
    blocks = list(get_code_blocks_from_paths(config, tmp_file_paths))
    expected_block = b'int calculate_power(int base, int exp) {\n    if (exp <= 0) {\n        return 1;\n    }\n    return base * calculate_power(base, exp - 1);\n}'
    assert len(blocks) == 1
    assert blocks[0][0] == expected_block
    assert blocks[0][1].type == "function_definition"
    assert blocks[0][1].start_point == (1, 0)
    assert blocks[0][1].end_point == (6, 1)


def test_get_code_blocks_from_paths_skips_non_utf8(config):
    tmp_file_path = config.workspace_path / "non_utf8_file.c"
    tmp_file_path.write_bytes(b"\xff\xfe not utf8 \x80")  # \xff\xfe = UTF-16 BOM
    tmp_file_paths = [tmp_file_path]
    blocks = list(get_code_blocks_from_paths(config, tmp_file_paths))
    assert blocks == []


def test_get_code_blocks_from_paths_returns_empty_for_no_files(config):
    blocks = list(get_code_blocks_from_paths(config, []))
    assert blocks == []


# --- get_code_blocks_from_auto_split ---

def test_get_code_blocks_from_auto_split(config, mocker):
    mocker.patch("codefinetuner.preprocess.extract._log_split_paths")
    config.raw_data_path = tests_path / "data"
    config.train_ratio = 0.4
    config.eval_ratio = 0.3
    config.test_ratio = 0.3
    split_result = get_code_blocks_from_auto_split(config)
    train_blocks = list(split_result.train_iter)
    eval_blocks = list(split_result.eval_iter)
    test_blocks = list(split_result.test_iter)
    assert len(train_blocks) == 5
    assert len(eval_blocks) == 5
    assert len(test_blocks) == 5


# --- _check_required_directories ---

def test_check_required_directories_passes_when_dirs_exist(config):
    config.raw_data_path.mkdir(parents=True, exist_ok=True)
    
    required_dirs = ["train", "eval", "test"]
    for dir_name in required_dirs:
        (config.raw_data_path / dir_name).mkdir(exist_ok=True)
    
    result = _check_required_directories(config.raw_data_path, required_dirs)
    assert result is None


def test_check_required_directories_raises_when_dir_missing(config):
    missing_dir_target = config.workspace_path / "missing_root"
    with pytest.raises(FileNotFoundError, match="Missing directories"):
        _check_required_directories(missing_dir_target, ["train", "eval", "test"])


# --- _get_filtered_paths ---

def test_get_filtered_paths_excludes_non_c_files(config):
    config.data_extensions = [".c"]
    tmp_readme_path = config.workspace_path / "test.py"
    tmp_readme_path.write_text("code", encoding="utf-8")
    tmp_cfile_path = config.workspace_path / "test.c"
    tmp_cfile_path.write_text("code", encoding="utf-8")
    filtered_paths = _get_filtered_paths(config, config.workspace_path)
    assert len(filtered_paths) == 1
    assert filtered_paths[0].suffix == ".c"


# --- get_code_blocks_from_manual_split ---

def test_get_code_blocks_from_manual_split_returns_three_iterators(config, mocker):
    mocker.patch("codefinetuner.preprocess.extract._log_split_paths")
    config.raw_data_path = tests_path / "data"
    split_result = get_code_blocks_from_manual_split(config)
    train_blocks = list(split_result.train_iter)
    eval_blocks = list(split_result.eval_iter)
    test_blocks = list(split_result.test_iter)
    assert len(train_blocks) > 0
    assert len(eval_blocks) > 0
    assert len(test_blocks) > 0


def test_get_code_blocks_from_manual_split_raises_without_dirs(config, tmp_path, mocker):
    mocker.patch("codefinetuner.preprocess.extract._log_split_paths")
    config.raw_data_path = tmp_path  # empty, no train/eval/test subdirs
    with pytest.raises(FileNotFoundError):
        get_code_blocks_from_manual_split(config)
