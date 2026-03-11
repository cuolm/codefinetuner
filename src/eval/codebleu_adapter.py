
# This adapter patches CodeBLEU so it uses tree-sitter-language-pack instead
# of importing tree_sitter_<lang> modules directly. This allows CodeBLEU to
# work with the project's existing tree-sitter-language-pack setup.

from typing import Dict, List, Union
import tree_sitter_language_pack
import codebleu.utils
import codebleu.codebleu


def shim_get_tree_sitter_language(lang: str):
    try:
        return tree_sitter_language_pack.get_language(lang)
    except Exception as e:
        raise ImportError(
            f"Could not load language {lang} from tree_sitter_language_pack: {e}"
        ) from e


# Apply patch so all internal CodeBLEU calls use the language pack
codebleu.utils.get_tree_sitter_language = shim_get_tree_sitter_language
codebleu.codebleu.get_tree_sitter_language = shim_get_tree_sitter_language


def codebleu_score(
    references: Union[List[str], List[List[str]]],
    predictions: List[str],
    lang: str = "python",
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> Dict[str, float]:
    from codebleu import calc_codebleu
    return calc_codebleu(references, predictions, lang=lang, weights=weights)

