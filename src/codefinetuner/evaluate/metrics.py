import logging
import re
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rapidfuzz.distance import Levenshtein

from .codebleu_shim import codebleu_score
from .config import Config


logger = logging.getLogger(__name__)


def _codebleu_structure_valid(config: Config, reference: str) -> bool:
    """
    Checks if the reference code is structurally complex enough for CodeBLEU.
    Performs a self-match test focusing only on Syntax (AST) and Data-flow. 
    If the reference is too simple (e.g., import blocks, comment) 
    to build these structures, the example is skipped. 
    """
    # suppress root logger warnings that are raised if it is not possible to calculate all 4 metrics
    root_logger = logging.getLogger()
    original_level = root_logger.getEffectiveLevel() 
    root_logger.setLevel(logging.ERROR)
    
    try:
        test_weights = (0.0, 0.0, 0.5, 0.5) 
        result = codebleu_score([reference], [reference], 
                               lang=config.codebleu_language, 
                               weights=test_weights)
    
        syntax_valid = result.get('syntax_match_score', 0) > 0
        dataflow_valid = result.get('dataflow_match_score', 0) > 0
        return syntax_valid and dataflow_valid
    except Exception:
        return False
    finally:
        # restore logger
        root_logger.setLevel(original_level)


def get_codebleu(config: Config, reference: str, prediction: str) -> tuple[float, bool]:
    """
    CodeBLEU: Computes weighted combination of four different similarity metrics:
    1. N-gram Match: Standard surface-level text overlap.
    2. Weighted N-gram: Text overlap with priority on keywords (if, else, etc.).
    3. Syntax (AST): Structural similarity between Abstract Syntax Trees.
    4. Data-flow: Logical similarity based on variable dependencies and usage.

    Examples are skipped if structural components (AST/Data-flow) cannot be 
    extracted from the reference. This prevents the final average score from 
    being unfairly lowered by snippets that cannot be properly parsed, 
    ensuring a more accurate representation of model quality.

    Standard weights are: [0.25, 0.25, 0.25, 0.25].
    CodeBLEU = (codebleu_ngram_weight * ngram_score) + 
               (codebleu_weighted_ngram_weight * weighted_ngram_score) + 
               (codebleu_syntax_ast_weight * syntax_ast_score) + 
               (codebleu_dataflow_weight * dataflow_score)
    """
    if not _codebleu_structure_valid(config, reference):
        return (0.0, False)
        
    try:
        codebleu_algorithm_weights = (
            config.codebleu_ngram_weight, 
            config.codebleu_weighted_ngram_weight, 
            config.codebleu_syntax_ast_weight, 
            config.codebleu_dataflow_weight
        )
        
        result = codebleu_score(
            [reference], [prediction], 
            lang=config.codebleu_language, 
            weights=codebleu_algorithm_weights
        )
        
        return (float(result['codebleu']), True)
        
    except Exception as e:
        logger.warning(f"CodeBLEU calculation failed, returning (0.0, False): {e}")
        return (0.0, False)


def get_sentencebleu(config: Config, reference: str, prediction: str) -> float:
    """
    SentenceBLEU: Measures n-gram overlap between reference and prediction.
    It rewards matching sequences of words (1-4) and uses smoothing 
    (Method1: Adds a tiny epsilon to all n-gram counts) to prevent a total 
    0.0 score when long sequences (e.g. 4-grams) don't match exactly.
    """
    try:
        reference_tokens = word_tokenize(reference)
        prediction_tokens = word_tokenize(prediction)
        
        weights = (
            config.sentencebleu_ngram_weight_1, 
            config.sentencebleu_ngram_weight_2, 
            config.sentencebleu_ngram_weight_3, 
            config.sentencebleu_ngram_weight_4
        )
        
        smoothing = SmoothingFunction().method1
        
        score = sentence_bleu(
            [reference_tokens], 
            prediction_tokens, 
            weights=weights, 
            smoothing_function=smoothing
        )
        
        return float(score)
        
    except Exception as e:
        logger.warning(f"SentenceBLEU calculation failed, returning 0.0: {e}")
        return 0.0


def get_exact_match(reference: str, prediction: str) -> float:
    """
    Exact match is 1.0 if identical, 0.0 otherwise. Collapese all whitespaces.
    Answers: "Did the model predict the right code logic regardless of formatting?"
    """
    try:
        # re.sub(r'\s+', ' ', text.strip()): Collapses whitespace to compare logic regardless of formatting.
        ref_norm = re.sub(r'\s+', ' ', reference.strip())
        pred_norm = re.sub(r'\s+', ' ', prediction.strip())
        
        if ref_norm == pred_norm:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"Exact match calculation failed, returning 0.0: {e}")
        return 0.0


def get_line_match(config: Config, reference: str, prediction: str) -> float:
    """"
    Checks if the prediction matches the reference up to n lines.
    If the reference has fewer than n lines, it matches against the full reference.
    Ignores trailing whitespaces.
    Answers: "Did the model predict the right first n lines including structure?"
    """
    try:
        n = config.line_match_number_of_lines

        # line.rstrip(): Removes trailing whitespace while preserving leading indentation.
        ref_lines_stripped = []
        for line in reference.splitlines()[:n]:
            ref_lines_stripped.append(line.rstrip())

        pred_lines_stripped = []
        for line in prediction.splitlines()[:n]:
            pred_lines_stripped.append(line.rstrip())

        if not ref_lines_stripped:
            return 0.0

        # we can only match up to the actual number of lines of the reference 
        effective_n = min(len(ref_lines_stripped), n)

        if len(pred_lines_stripped) < effective_n:
            return 0.0

        if pred_lines_stripped[:effective_n] == ref_lines_stripped[:effective_n]:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"Line match calculation failed, returning 0.0: {e}")
        return 0.0


def get_edit_similarity(reference: str, prediction: str) -> float:
    """
    Edit Similarity: 1 - Levenshtein(prediction, reference) / max(len(prediction), len(reference))
    Returns 1.0 for identical strings, 0.0 for completely different strings.
    Answers: "How close is the prediction to the reference at the character level?"
    """
    try:
        if not reference and not prediction:
            return 1.0
        if not reference or not prediction:
            return 0.0

        return float(Levenshtein.normalized_similarity(prediction, reference))

    except Exception as e:
        logger.warning(f"Edit similarity calculation failed, returning 0.0: {e}")
        return 0.0
