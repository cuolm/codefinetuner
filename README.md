# CodeFinetuner
[![PyPI](https://img.shields.io/pypi/v/codefinetuner.svg)](https://pypi.org/project/codefinetuner/)
[![License](https://img.shields.io/github/license/cuolm/codefinetuner.svg)](LICENSE.txt)
[![Release](https://github.com/cuolm/codefinetuner/actions/workflows/release.yaml/badge.svg)](https://github.com/cuolm/codefinetuner/actions/workflows/release.yaml)
[![Tests](https://github.com/cuolm/codefinetuner/actions/workflows/tests.yaml/badge.svg)](https://github.com/cuolm/codefinetuner/actions/workflows/tests.yaml)

CodeFinetuner fine-tunes a local code autocomplete model on your own repository for use in editors like VS Code or Vim/Neovim.
It trains a Low-Rank Adapter ([LoRA](https://arxiv.org/abs/2106.09685)) on Fill-In-the-Middle ([FIM](https://arxiv.org/abs/2207.14255)) examples so the model learns the structure and patterns of your codebase.

## Table of Contents
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How Training Examples Are Created](#how-training-examples-are-created)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Finetuned Model Usage](#finetuned-model-usage)
- [Docker Image](#docker-image)
- [Tree-sitter Customization](#tree-sitter-customization)
- [Tests](#tests)
- [Resources](#resources)
- [License](#license)


## Architecture
CodeFinetuner follows a simple pipeline. First, raw code is parsed and turned into FIM examples. These examples are then used to train a LoRA adapter and evaluate the fine‑tuned model using multiple metrics. Finally, the model is converted into GGUF format for deployment.

```text
Raw Code Files
     |
     v
[Preprocess]  -- tree-sitter parsing -> FIM examples -> tokenized jsonl datasets
     |
     v
[Finetune]    -- LoRA adapter training -> merged safetensors model
     |
     v
[Evaluate]    -- CodeBLEU, SentenceBLEU, exact match, line match, perplexity
     |
     v
[Convert]      -- GGUF conversion -> quantized model for deployment
```

## Project Structure
```text
.
├── src/
│   └── codefinetuner/           # Core packages
│       ├── preprocess/
│       ├── finetune/
│       ├── evaluate/
│       └── convert/             
├── config/                      # User configuration
│   └── codefinetuner_config.yaml
├── data/                        # Default data directory 
├── outputs/                     # Pipeline outputs
├── scripts/                     # Utility scripts
├── tests/                       # Tests 
├── third_party/                 # External submodules 
└── docs/                        # Documentation 
```

## How Training Examples Are Created
CodeFinetuner builds FIM examples from real code structure, not random text chunks. It first extracts blocks such as functions or classes, then masks smaller sub-blocks like statements or expressions for the model to predict. This approach helps the model learn the logical structure of your codebase instead of unrelated fragments.

Here is an example illustrating how a single FIM example is created:
<table>
  <tr>
    <td align="center" valign="top">
      <strong>Source Code File</strong><br>
      <img src="/docs/code_file.png" alt="code_file" width="250">
    </td>
    <td align="center" valign="top">
      <strong>Code Block</strong><br>
      <img src="/docs/code_block.png" alt="code_block" width="250">
    </td>
    <td align="center" valign="top">
      <strong>One Subblock</strong><br>
      <img src="/docs/code_subblock.png" alt="code_subblock" width="250">
    </td>
  </tr>
</table>

```python
<|fim_prefix|>uint32_t count_bits(uint32_t value){\n  uint32_t count = 0;\n  while(value){\n    
<|fim_suffix|>    }\n    return count;
<|fim_middle|>count = count + (value & 1);\n    value = (value >> 1);
```

## Installation

### From PyPI
```bash
uv add codefinetuner
# or
pip install codefinetuner
```
### From Source
```bash
git clone --recurse-submodules https://github.com/cuolm/codefinetuner
cd codefinetuner

# Using uv (Recommended)
uv sync

# Using pip
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start
Create a configuration file according to the [Configuration](#configuration) section.
```python
import codefinetuner

codefinetuner.run_pipeline("codefinetuner_config.yaml")
```

## Configuration
The pipeline uses a single-source-of-truth YAML configuration file. It utilizes YAML anchors (`&globals`) to share core parameters across all stages (`preprocess`, `finetune`, `evaluate`, `convert`), ensuring consistency and reducing redundancy.

### Configuration Structure

Create `codefinetuner_config.yaml` using the template below. For the full parameter list, see the [Configuration Reference Guide](/docs/config-file.md).

```yaml
# globals contain all the mandatory parameters.
globals: &globals
  workspace_path: null  # null: defaults to current working directory (CWD)
  model_name: "Qwen/Qwen2.5-Coder-1.5B" 
  fim_prefix_token: "<|fim_prefix|>"
  fim_middle_token: "<|fim_middle|>"
  fim_suffix_token: "<|fim_suffix|>"
  fim_pad_token: "<|fim_pad|>"
  eos_token: "<|endoftext|>"
  label_pad_token_id: -100
  data_language: "c"
  data_extensions: [".c", ".h"]

preprocess:
  <<: *globals                   # Inherits all global parameters
  split_mode: "manual"
  max_token_sequence_length: 1024
  # ... (preprocess specific settings)

finetune:
  <<: *globals
  lora_r: 32
  trainer_num_train_epochs: 1
  # ... (finetune specific settings)

evaluate:
  <<: *globals
  benchmark_sample_size: 4
  # ... (evaluate specific settings)

convert:
  <<: *globals
  # ... (convert specific settings)
```
> **Note:** See [`config/codefinetuner_config.yaml`](/config/codefinetuner_config.yaml) for a full production example.

### Data Preparation
Place source files in your `raw_data_path` (default: `workspace_path/data`).
* **Auto Split:** Place files directly in the directory.
* **Manual Split:** Create `train`, `eval`, and `test` subfolders inside `raw_data_path` and assign files according to your manual split preferences.

## Usage

### CLI Usage
Run the pipeline using the unified CLI:
```bash
uv run codefinetuner --config="config/codefinetuner_config.yaml"
```

**Pipeline flags**
- `--config`: Use a different config file.
- `--skip-preprocess`: Skip preprocessing.
- `--skip-finetune`: Skip fine-tuning.
- `--skip-evaluate`: Skip evaluation.
- `--skip-convert`: Skip conversion.

### Python Module Usage
```python
import codefinetuner

# Full pipeline
codefinetuner.run_pipeline("path/to/codefinetuner_config.yaml")

# Skip stages
codefinetuner.run_pipeline(
    "path/to/codefinetuner_config.yaml",
    skip_preprocess=True,
    skip_convert=True
)
```

## Finetuned Model Usage
The `convert` stage exports the final model to [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) format for local inference. The resulting file is saved at `outputs/convert/results/lora_model.gguf`. 
For setup instructions with the VS Code extension [llama.vscode](https://github.com/ggml-org/llama.vscode), see the [inference-vscode](/docs/inference-vscode.md) guide.


## Docker Image

#### 1. Build the Docker Image
Build the image from the `Dockerfile`, tagging it as codefinetuner-image.
```bash
docker build -t codefinetuner-image .
```
#### 2. Prepare Data and Run the Container
To allow the container to access your data for fine-tuning, use a bind mount to link your host machine's `data` directory to the container.  
On your host machine (where you run Docker), create a folder named `data` if it does not already exist. Put all files you want to use for fine-tuning inside the `data` directory. For manual mode, include `train`, `eval`, and `test` subfolders with the split you want to use.  

#### NVIDIA GPU (Recommended)
Use this command to enable CUDA support for `torch` and `bitsandbytes`. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host machine.

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/data:/app/data \
  codefinetuner-image /bin/bash
```
#### CPU Only
Use this if you do not have a compatible GPU. Fine-tuning will be much slower.
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  codefinetuner-image /bin/bash
```

## Tree-sitter Customization
Tree-sitter turns source code into structural blocks used to generate FIM examples. Use this section to add new languages or build missing parsers.

- [Add Language Definitions](/docs/tree-sitter-customization.md#add-new-language-block-definitions): define `block_types` and `subblock_types` in JSON.
- [Build Custom Parser](/docs/tree-sitter-customization.md#build-custom-parser): compile a parser from source, for example for Mojo.

## Tests
Run the test suite with:

```bash
pytest
```

## Resources
- [Qwen2.5-Coder Technical Report](https://arxiv.org/pdf/2409.12186)
- [Structure-Aware Fill-in-the-Middle Pretraining for Code](https://arxiv.org/pdf/2506.00204)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/pdf/2207.14255)
- [From Output to Evaluation: Does Raw Instruction-Tuned Code LLMs Output Suffice for Fill-in-the-Middle Code Generation?](https://arxiv.org/pdf/2505.18789)
- [CodeBLEU: a Method for Automatic Evaluation of Code Synthesis](https://arxiv.org/pdf/2009.10297)
- [HF LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [llama.vscode](https://github.com/ggml-org/llama.vscode)

## License
Licensed under the [Apache License 2.0](/LICENSE.txt).