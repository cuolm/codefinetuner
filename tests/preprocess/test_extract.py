import pathlib
import textwrap

import pytest


tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.preprocess.config import Config
from codefinetuner.preprocess.extract import (
    auto_create_split_paths,
    _extract_code_blocks_rec,
    get_code_blocks_from_paths,
    get_code_blocks_from_auto_split,
    _check_required_directories,
    _get_filtered_paths,
    get_code_blocks_from_manual_split
)


# --- Fixtures ---

@pytest.fixture
def config() -> Config:
    """Load a Config from the test YAML, redirecting outputs to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    return test_config


# --- auto_create_split_paths ---

def test_auto_create_split_paths_partitions_all_files(tmp_path):
    """auto split on a flat directory should account for all files."""
    tmp_data_path = tmp_path / "data"
    tmp_data_path.mkdir()
    num_files = 10
    for i in range(num_files):
        tmp_file = tmp_data_path / f"file_{i}.c"
        tmp_file.write_text(f"int f{i}(){{}}", encoding="utf-8")

    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config.raw_data_path = tmp_data_path 
    test_config.split_mode = "auto"
    test_config.train_ratio = 0.7
    test_config.eval_ratio = 0.2
    test_config.test_ratio = 0.1
    test_config._setup_paths()
    test_config._ensure_output_paths_exist()

    train, eval, test = auto_create_split_paths(test_config)
    assert len(train) == 7
    assert len(eval) == 2
    assert len(test) == 1


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

def test_get_code_blocks_from_paths_yields_correct_blocks(config, tmp_path):
    tmp_cfile_path = tmp_path / "test_file.c"
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


def test_get_code_blocks_from_paths_skips_non_utf8(config, tmp_path):
    tmp_file_path = tmp_path / "non_utf8_file.c"
    tmp_file_path.write_bytes(b"\xff\xfe not utf8 \x80")  # \xff\xfe = UTF-16 BOM
    tmp_file_paths = [tmp_file_path]
    blocks = list(get_code_blocks_from_paths(config, tmp_file_paths))
    assert blocks == []


def test_get_code_blocks_from_paths_returns_empty_for_no_files(config):
    blocks = list(get_code_blocks_from_paths(config, []))
    assert blocks == []


# --- get_code_blocks_from_auto_split ---

def test_get_code_blocks_from_auto_split(config):
    config.train_ratio = 0.4
    config.eval_ratio = 0.3
    config.test_ratio = 0.3
    train_iter, eval_iter, test_iter = get_code_blocks_from_auto_split(config)
    train_blocks = list(train_iter)
    eval_blocks = list(eval_iter)
    test_blocks = list(test_iter)
    assert len(train_blocks) == 5
    assert len(eval_blocks) == 5
    assert len(test_blocks) == 5


# --- _check_required_directories ---

def test_check_required_directories_passes_when_dirs_exist(config):
    _check_required_directories(config.raw_data_path, ["train", "eval", "test"])


def test_check_required_directories_raises_when_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing directories"):
        _check_required_directories(tmp_path, ["train", "eval", "test"])


# --- _get_filtered_paths ---

def test_get_filtered_paths_excludes_non_c_files(config, tmp_path):
    tmp_readme_path = tmp_path / "test.py"
    tmp_readme_path.write_text("code", encoding="utf-8")
    tmp_cfile_path = tmp_path / "test.c"
    tmp_cfile_path.write_text("code", encoding="utf-8")
    filtered_paths = _get_filtered_paths(config, tmp_path)
    assert len(filtered_paths) == 1
    assert filtered_paths[0].suffix == ".c" 


# --- get_code_blocks_from_manual_split ---

def test_get_code_blocks_from_manual_split_returns_three_iterators(config):
    train_iter, eval_iter, test_iter = get_code_blocks_from_manual_split(config)
    train_blocks = list(train_iter)
    eval_blocks = list(eval_iter)
    test_blocks = list(test_iter)
    assert len(train_blocks) > 0
    assert len(eval_blocks) > 0
    assert len(test_blocks) > 0


def test_get_code_blocks_from_manual_split_raises_without_dirs(config, tmp_path):
    config.raw_data_path = tmp_path  # empty, no train/eval/test subdirs
    with pytest.raises(FileNotFoundError):
        get_code_blocks_from_manual_split(config)
