from dataclasses import dataclass, field
from pathlib import Path

import nltk
import torch

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    fim_prefix_token: str = "<|fim_prefix|>"
    fim_suffix_token: str = "<|fim_suffix|>"
    fim_middle_token: str = "<|fim_middle|>"
    fim_pad_token: str = "<|fim_pad|>"
    input_sample_size: int = 2 
    min_fim_middle_chars: int = 0 

    prepare_benchmark_shuffle_buffer_size: int = 10000000
    prepare_benchmark_shuffle_seed: int = 42 

    gen_max_new_tokens: int = 128 
    gen_do_sample: bool = False  # note: if set to False temperature and top_p have no effect
    gen_temperature: float = 0.7
    gen_top_p: float = 0.95

    project_root_path: Path = field(init=False)
    trainer_output_dir_path: Path = field(init=False)
    input_dataset_path: Path = field(init=False)  # tokenized dataset under /datasets e.g. test_dataset.jsonl
    benchmark_dataset_path: Path = field(init=False)
    base_results_tmp_path: Path = field(init=False)
    evaluation_results_path: Path = field(init=False)
    plot_file: Path = field(init=False)
    device: str = field(init=False)

    # CodeBLEU weights (must sum to 1.0)
    # see: https://arxiv.org/pdf/2009.10297  Section 4.4 for parameter suggestions 0.1, 0.1, 0.4, 0.4
    codebleu_language: str = "c"
    codebleu_score_name: str = "codebleu"
    codebleu_ngram_weight: float = 0.25         # token-level overlap (standard BLEU)
    codebleu_weighted_ngram_weight: float = 0.25 # keyword-level overlap (importance-weighted)
    codebleu_syntax_ast_weight: float = 0.25     # structural correctness (Abstract Syntax Tree)
    codebleu_dataflow_weight: float = 0.25       # logic consistency (Variable dependency graph)
    codebleu_plot_file: Path = field(init=False)

    # sentence-BLEU weights (must sum to 1.0)
    sentencebleu_score_name: str = "sentencebleu"
    sentencebleu_ngram_weight_1: float = 0.25  # 1-gram
    sentencebleu_ngram_weight_2: float = 0.25  # 2-gram  
    sentencebleu_ngram_weight_3: float = 0.25  # 3-gram
    sentencebleu_ngram_weight_4: float = 0.25  # 4-gram
    sentencebleu_plot_file: Path = field(init=False)

    exact_match_score_name: str = "exact_match"
    exact_match_plot_file: Path = field(init=False)

    line_match_score_name: str = "line_match"
    line_match_number_of_lines: int = 2  # 5 is standard production value (Sourcegraph, Cursor IDE)
    line_match_plot_file: Path = field(init=False)

    perplexity_name: str = "perplexity"
    perplexity_plot_file: Path = field(init=False)

    _nltk_initialized: bool = field(init=False, default=False)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        cb_total = (self.codebleu_ngram_weight + self.codebleu_weighted_ngram_weight +
            self.codebleu_syntax_ast_weight + self.codebleu_dataflow_weight)
        if abs(cb_total - 1.0) > 1e-6:
            raise ValueError(f"CodeBLEU weights must sum to 1.0, got {cb_total}")

        self.project_root_path = Path(__file__).resolve().parent.parent.parent
        self.trainer_output_dir_path = self.project_root_path / "results"
        self.input_dataset_path = self.project_root_path / "datasets" / "test_dataset.jsonl"
        self.benchmark_dataset_path = self.project_root_path / "benchmarks" / "pebble_test_examples.jsonl"
        self.base_results_tmp_path = self.project_root_path / "benchmarks" / "results" / "base_results_tmp.jsonl"
        self.evaluation_results_path = self.project_root_path / "benchmarks" / "results" / "evaluation_results.jsonl"
        self.codebleu_plot_file = self.project_root_path / "benchmarks" / "results" / "codebleu_plot.png"
        self.sentencebleu_plot_file = self.project_root_path / "benchmarks" / "results" / "sentencebleu_plot.png"
        self.exact_match_plot_file = self.project_root_path / "benchmarks" / "results" / "exact_match_plot.png"
        self.line_match_plot_file = self.project_root_path / "benchmarks" / "results" / "line_match_plot.png"
        self.perplexity_plot_file = self.project_root_path / "benchmarks" / "results" / "perplexity_plot.png"
        self.evaluation_report_path = self.project_root_path / "benchmarks" / "results" / "evaluation_report.json"

        self.all_metrics_average = self.project_root_path / "benchmarks" / "results" / "all_metrics_average.png"
    
    @property 
    def nltk_ready(self) -> bool:
        if not self._nltk_initialized:
            nltk.download('punkt', quiet=True)  
            nltk.download('punkt_tab', quiet=True)
            self._nltk_initialized = True
        return True
