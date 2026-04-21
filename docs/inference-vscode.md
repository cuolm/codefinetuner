# VS Code Inference Guide

## How to Use the LoRA Fine-tuned Model
You can use the fine-tuned model with any code autocompletion tool that supports FIM (Fill-In-the-Middle) models. This section explains how to use the VS Code extension [llama.vscode](https://github.com/ggml-org/llama.vscode) to run the model locally on your machine.

### 1. Install llama.vscode
Install the extension from the VS Code Marketplace: [Click here](https://marketplace.visualstudio.com/items?itemName=ggml-org.llama-vscode).

### 2. Ensure llama.cpp is Installed
Open the **llama-vscode** menu by clicking it in the status bar or pressing `Ctrl+Shift+M`, then select **Install/Upgrade llama.cpp**.

*Note: On macOS, you can also install via Homebrew:*
```bash
brew install llama.cpp
```

### 3. Select the Fine-tuned Model in llama-vscode
In the llama-vscode menu:  
-> "Completion models..."  
-> "Add local completion model..."  
-> Enter model name lora_model_gguf (or any name you prefer).  
-> Enter the following command to start the model locally:  
```bash
    llama-server -m /path/to/your/codefinetuner_workspace/outputs/convert/results/lora_model.gguf --port 8012
```  
-> Confirm Endpoint ```http://127.0.0.1:8012```  
-> No API Key required

Then, in the llama-vscode menu:  
->"Completion models..."  
->"Select/Start completion model..."  
->"lora_model_gguf"