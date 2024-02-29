import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
import argparse
import itertools

def read_text_file(file_path):
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        return text
    except ImportError as e:
        raise ImportError(f"Error importing PyMuPDF (fitz): {e}")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")

def get_files_from_directory(directory, file_extension):
    files = []
    try:
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(file_extension):
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
    except Exception as e:
        print(f"Error while getting files: {e}")
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained model on a text classification task.")
    parser.add_argument("-d", "--directoryname", help="Path to the input files directory", required=True)
    parser.add_argument("-m", "--modelname", help="Name of the pre-trained model", required=True)
    parser.add_argument("-e", "--numberofepochs", type=int, help="Number of training epochs", default=3)
    parser.add_argument("-o", "--outputfilename", help="Output filename for the fine-tuned model", required=True)
    parser.add_argument("-x", "--fileextension", help="File extension to consider (e.g., pdf)", default=".pdf")
    parser.add_argument("-g", "--gpuid", type=int, help="GPU ID to use for training", default=-1)
    parser.add_argument("-b", "--batchamount", type=int, help="Number of batches", default=128)
    parser.add_argument("-l", "--learningrateinscientificnotation", type=float, help="Learning rate in scientific notation", default=3e-4)
    parser.add_argument("-s", "--stopatloss", type=float, help="Stop training when loss falls below this value", default=0)
    parser.add_argument("-mb", "--microbatchsize", type=int, help="Micro Batch Size", default=4)
    parser.add_argument("-c", "--cutofflength", type=int, help="Cutoff length", default=256)
    args = parser.parse_args()

    # Set GPU if specified
    if args.gpuid >= 0:
        torch.cuda.set_device(args.gpuid)

    input_directory = args.directoryname
    model_name = args.modelname
    epochs = args.numberofepochs
    output_filename = args.outputfilename
    file_extension = args.fileextension
    batch_amount = args.batchamount
    learning_rate = args.learningrateinscientificnotation
    stop_at_loss = args.stopatloss
    micro_batch_size = args.microbatchsize
    cutoff_length = args.cutofflength

    static_directory = os.path.join(input_directory, 'static')
    os.makedirs(static_directory, exist_ok=True)

    # Get a list of files in the specified directory with the specified extension
    input_files = get_files_from_directory(input_directory, file_extension)

    if not input_files:
        print(f"No {file_extension} files found in the specified directory: {input_directory}")
        exit()

    # Load your specified pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Move model to GPU if specified
    if args.gpuid >= 0:
        model = model.to("cuda")

    # Set up optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()

        for input_filename in input_files:
            # Read the contents of the file
            try:
                input_text = read_text_file(input_filename)
            except Exception as e:
                print(f"Error reading file {input_filename}: {e}")
                continue

            # Tokenize and encode the input text
            tokenized_inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt", max_length=cutoff_length)

            # Dummy label (replace with your actual label logic)
            label = torch.tensor([0], dtype=torch.float32)  # Convert label to float

            # Move tensors to GPU if specified
            if args.gpuid >= 0:
                tokenized_inputs = {key: value.to("cuda") for key, value in tokenized_inputs.items()}
                label = label.to("cuda")

            # Create a DataLoader for the single example
            dataset = TensorDataset(
                tokenized_inputs['input_ids'],
                tokenized_inputs['attention_mask'],
                label
            )
            dataloader = DataLoader(dataset, batch_size=batch_amount, shuffle=False)

            # Micro Batching
            for i in range(0, len(dataloader), micro_batch_size):
                micro_batch = list(itertools.islice(dataloader, i, i + micro_batch_size))

                # Training step
                for batch in micro_batch:
                    optimizer.zero_grad()
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                    outputs = model(**inputs, labels=batch[2])
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    # Check if loss falls below stop_at_loss
                    if loss.item() < stop_at_loss:
                        print(f"Training stopped as loss reached {loss.item()} which is below the specified stop_at_loss {stop_at_loss}")
                        break

    # Save the fine-tuned model with the specified output filename and format
    model.save_pretrained(output_filename, save_format='safetensors')

