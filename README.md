# Tools for Tuning Embedding Models 
This is a repository of tools that can be used for tuning embedding models (not Large Language Models). There are two tools included, one that uses Torch and Transformers and one that uses Llamaindex. 

## torchTune.py 
### Description
torchTune.py is a script for fine-tuning pre-trained models on a specific text classification task. It leverages the Hugging Face Transformers library to load and fine-tune models for sequence classification. The script supports processing input files (e.g., PDFs) from a specified directory, tokenizing the text content, and performing the training loop with specified hyperparameters.

### Usage
To use torchTune.py, you need to specify several command-line arguments detailing the input files, model configuration, and training parameters. Here's how to invoke the script:

```
python3.11 torchTune.py -m <modelname> -d <directoryname> -o <outputfilename> [options]
```
- -m, --modelname: Name of the pre-trained model to fine-tune.
- -d, --directoryname: Path to the input files directory.
- -o, --outputfilename: Output filename for the fine-tuned model.
- -e, --numberofepochs: Number of training epochs (default: 3).
- -x, --fileextension: File extension to consider (default: .pdf).
- -g, --gpuid: GPU ID to use for training (default: -1 for CPU).
- -b, --batchamount: Number of batches (default: 128).
- -l, --learningrateinscientificnotation: Learning rate in scientific notation (default: 3e-4).
- -s, --stopatloss: Stop training when loss falls below this value (default: 0).
- -mb, --microbatchsize: Micro Batch Size (default: 4).
- -c, --cutofflength: Cutoff length for tokenization (default: 256).

Example:
```
python3.11 torchTune.py -m bge-large-en-v1.5 -d /path/to/input/dir -o torch
```

## LlamaIndexTune.py
Description
llamaIndexTune.py is designed for fine-tuning and evaluating embedding models for information retrieval tasks. It uses a combination of tools and libraries, including llama_index and sentence_transformers, to process documents, generate question-answer pairs for training, and evaluate the performance of the fine-tuned embedding model.

### Usage
This script requires specifying the model and the directory containing the files to process. Here's the syntax for running llamaIndexTune.py:

```
python3.11 llamaIndexTune.py -m <modelname> -d <directoryname> [options]
```
- -m, --modelname: Name of the embedding model to fine-tune.
- -d, --directoryname: Name of the directory to search for files.

Example:
```
python3.11 llamaIndexTune.py -m bge-large-en-v1.5 -d /path/to/directory
```
## Requirements
Both tools require Python 3.11 and specific Python packages, which can be installed via pip. Ensure you have the following packages installed:

- transformers
- torch
- PyMuPDF (for torchTune.py)
- llama_index (for llamaIndexTune.py)
- sentence_transformers (for llamaIndexTune.py)

To install these packages, you can use the following command:
```
pip3.11 install -r requirements.txt
```





