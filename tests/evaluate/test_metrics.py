import pytest
import pathlib

from codefinetuner.evaluate.config import Config
import codefinetuner.evaluate.metrics as metrics
from codefinetuner.evaluate.metrics import (
    _codebleu_language_supported,
    _codebleu_structure_valid,
    get_codebleu,
    _ensure_nltk_initialized,
    get_sentencebleu,
    get_exact_match,
    get_line_match,
    get_edit_similarity
)

tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> Config:
    """Load an evaluate Config from the test YAML."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config._setup_paths()
    return test_config 


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


# --- _codebleu_language_supported ---

def test_codebleu_language_supported_true(config):
    config.data_language = "c"
    assert _codebleu_language_supported(config) is True


def test_codebleu_language_supported_false(config):
    config.data_language = "iec61131_3_st"
    assert _codebleu_language_supported(config) is False


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
    mocker.patch("codefinetuner.evaluate.metrics._codebleu_language_supported", return_value=True)
    mocker.patch("codefinetuner.evaluate.metrics._codebleu_structure_valid", return_value=True)
    mocker.patch("codefinetuner.evaluate.metrics.codebleu_score", return_value={'codebleu': 0.85})
    
    score, valid = get_codebleu(config, "ref", "pred")
    assert score == 0.85
    assert valid is True


def test_get_codebleu_exception(config, mocker):
    mocker.patch("codefinetuner.evaluate.metrics._codebleu_language_supported", return_value=True)
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

def test_get_codebleu_unsupported_language_short_circuits(config, mocker):
    config.data_language = "iec61131_3_st"
    structure_mock = mocker.patch("codefinetuner.evaluate.metrics._codebleu_structure_valid", return_value=True)

    score, valid = get_codebleu(config, "ref", "pred")
    assert score == 0.0
    assert valid is False
    structure_mock.assert_not_called()


# --- _ensure_nltk_initialized ---

def test_ensure_nltk_initialized_skips_if_already_done(config, mocker):
    download_mock = mocker.patch("nltk.download")
    mocker.patch("codefinetuner.evaluate.metrics._NLTK_INITIALIZED", True)

    _ensure_nltk_initialized()
    download_mock.assert_not_called()


def test_ensure_nltk_initialized_runs_successfully(config, mocker):
    download_mock = mocker.patch("nltk.download")
    mocker.patch("codefinetuner.evaluate.metrics._NLTK_INITIALIZED", False)

    _ensure_nltk_initialized()
    assert download_mock.call_count == 2
    assert metrics._NLTK_INITIALIZED is True


def test_ensure_nltk_initialized_raises_runtime_error_on_failure(config, mocker):
    mocker.patch("nltk.download", side_effect=Exception("Network connection timeout"))
    mocker.patch("codefinetuner.evaluate.metrics._NLTK_INITIALIZED", False)

    with pytest.raises(RuntimeError, match="NLTK initializerion failed"):
        _ensure_nltk_initialized() 


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


# --- get_edit_similarity ---

def test_get_edit_similarity_identical():
    score = get_edit_similarity("line", "line")
    assert score == 1.0


def test_get_edit_similarity_both_empty():
    score = get_edit_similarity("", "")
    assert score == 1.0


def test_get_edit_similarity_reference_empty():
    score = get_edit_similarity("", "some code")
    assert score == 0.0


def test_get_edit_similarity_prediction_empty():
    score = get_edit_similarity("some code", "")
    assert score == 0.0


def test_get_edit_similarity_exception(mocker):
    mocker.patch("codefinetuner.evaluate.metrics.Levenshtein.normalized_similarity", side_effect=Exception("Simulated error"))
    score = get_edit_similarity("ref", "pred")
    assert score == 0.0


def test_get_edit_similarity_integration_high_score():
    score = get_edit_similarity(C_FUNCTION, C_FUNCTION)
    assert score == 1.0


def test_get_edit_similarity_integration_low_score():
    score = get_edit_similarity(C_FUNCTION, C_VARIABLE)
    assert score < 0.3
