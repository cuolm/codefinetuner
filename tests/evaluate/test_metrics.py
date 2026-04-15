import pytest
import pathlib

from codefinetuner.evaluate.config import Config
from codefinetuner.evaluate.metrics import (
    _codebleu_structure_valid,
    get_codebleu,
    get_sentencebleu,
    get_exact_match,
    get_line_match
)

tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

# --- Fixtures ---

@pytest.fixture
def config() -> Config:
    """Load an evaluate Config from the test YAML."""
    return Config.load_from_yaml(test_config_path)


# --- Constants ---

C_FUNCTION = (
    "int calculate_sum(int n) {\n"
    "    if (n <= 0) {\n"
    "        return 0;\n"
    "    }\n"
    "    return n + calculate_sum(n - 1);\n"
    "}"
)

C_VARIABLE = (
    "int i = 0;"
)


# --- _codebleu_structure_valid ---

def test_codebleu_structure_valid_true(config, mocker):
    mock_score = mocker.patch("codefinetuner.evaluate.metrics.codebleu_score", 
                              return_value={'syntax_match_score': 1.0, 'dataflow_match_score': 1.0})
    assert _codebleu_structure_valid(config, "valid code") is True


def test_codebleu_structure_valid_false(config, mocker):
    mocker.patch("codefinetuner.evaluate.metrics.codebleu_score", 
                 return_value={'syntax_match_score': 0.0, 'dataflow_match_score': 1.0})
    assert _codebleu_structure_valid(config, "invalid code") is False


def test_codebleu_structure_valid_integration_true(config):
    assert _codebleu_structure_valid(config, C_FUNCTION) is True


def test_codebleu_structure_valid_integration_false(config):
    """ C_VARIABLE not complex enought to calc codebleu"""
    assert _codebleu_structure_valid(config, C_VARIABLE) is False   


# --- get_codebleu ---

def test_get_codebleu_passes(config, mocker):
    mocker.patch("codefinetuner.evaluate.metrics._codebleu_structure_valid", return_value=True)
    mocker.patch("codefinetuner.evaluate.metrics.codebleu_score", return_value={'codebleu': 0.85})
    
    score, valid = get_codebleu(config, "ref", "pred")
    assert score == 0.85
    assert valid is True


def test_get_codebleu_exception(config, mocker):
    mocker.patch("codefinetuner.evaluate.metrics._codebleu_structure_valid", return_value=True)
    mocker.patch("codefinetuner.evaluate.metrics.codebleu_score", side_effect=Exception("Simulated codebleu error"))

    score, valid = get_codebleu(config, "ref", "pred")
    assert score == 0.0
    assert valid is False



def test_get_codebleu_invalid_structure(config, mocker):
    mocker.patch("codefinetuner.evaluate.metrics._codebleu_structure_valid", return_value=False)
    
    score, valid = get_codebleu(config, "ref", "pred")
    assert score == 0.0
    assert valid is False


def test_get_codebleu_integration_passes(config):
    score, valid = get_codebleu(config, C_FUNCTION, C_FUNCTION)
    assert valid is True
    assert score > 0.9


def test_get_codebleu_integration_calculation_failed(config):
    """ C_VARIABLE not complex enough to calculate codebleu """
    score, valid = get_codebleu(config, C_VARIABLE, C_VARIABLE)
    assert valid is False 
    assert score == 0.0 


# --- get_sentencebleu ---

def test_get_sentencebleu_passes(config, mocker):
    mocker.patch("codefinetuner.evaluate.metrics.word_tokenize", side_effect=lambda x: x.split())
    mocker.patch("codefinetuner.evaluate.metrics.sentence_bleu", return_value=0.75)
    
    score = get_sentencebleu(config, "ref", "pred")
    assert score == 0.75


def test_get_sentencebleu_exception(config, mocker):
    mocker.patch("codefinetuner.evaluate.metrics.word_tokenize", side_effect=Exception("Simulated sentencebleu error"))
    score = get_sentencebleu(config, "ref", "pred") 
    assert score == 0.0


def test_get_sentencebleu_integration_high_score(config):
    score = get_sentencebleu(config, C_FUNCTION, C_FUNCTION)
    assert score > 0.9 


def test_get_sentencebleu_integration_low_score(config):
    score = get_sentencebleu(config, C_FUNCTION, C_VARIABLE)
    assert score < 0.1


# --- get_exact_match ---

def test_get_exact_match_identical():
    score = get_exact_match("def add(): pass", "def add(): pass") 
    assert score == 1.0


def test_get_exact_match_whitespace_normalization():
    score = get_exact_match("def add(): pass", "  def add():\n pass  ") 
    assert score == 1.0


def test_get_exact_match_different():
    score = get_exact_match("reference_different", "prediction_different")
    assert score == 0.0


def test_get_exact_match_exception(mocker):
    mocker.patch("codefinetuner.evaluate.metrics.logger.warning")
    score = get_exact_match(None, "def add(): pass")  # passing None triggers AttributeError on None.strip()
    assert score == 0.0


# --- get_line_match ---

def test_get_line_match_identical(config):
    config.line_match_number_of_lines = 2
    reference = "line1\nline2"
    prediction = "line1\nline2" 
    score = get_line_match(config, reference, prediction) 
    assert  score == 1.0


def test_get_line_match_first_n_lines_match(config):
    config.line_match_number_of_lines = 2
    reference = "line1\nline2\nline3"
    prediction = "line1\nline2\nextra line"
    score = get_line_match(config, reference, prediction)
    assert score == 1.0


def test_get_line_match_ignores_trailing_whitespace_on_lines(config):
    config.line_match_number_of_lines = 2
    reference = "line1   \nline2"
    prediction = "line1\nline2"
    score = get_line_match(config, reference, prediction)
    assert score == 1.0


def test_get_line_match_mismatch(config):
    config.line_match_number_of_lines = 1
    reference = "line"
    prediction = "different_line"
    score = get_line_match(config, reference, prediction) 
    assert score == 0.0


def test_get_line_match_first_line_mismatch(config):
    config.line_match_number_of_lines = 2
    reference = "line1\nline2"
    prediction = "different\nline2"
    score = get_line_match(config, reference, prediction)
    assert score == 0.0


def test_get_line_match_prediction_too_short_returns_zero(config):
    config.line_match_number_of_lines = 2
    reference = "line1\nline2"
    prediction = "line1"
    score = get_line_match(config, reference, prediction)
    assert score == 0.0
