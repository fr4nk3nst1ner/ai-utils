import json
import argparse
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SentenceSplitter
from llama_index.schema import MetadataMode
from llama_index.finetuning import generate_qa_embedding_pairs, EmbeddingQAFinetuneDataset
from llama_index.llms import OpenAI  # Replace with your custom class or function
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd
import os
from tqdm.notebook import tqdm
from pathlib import Path

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes

def evaluate(dataset, embed_model, top_k=5, verbose=False):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(
        nodes, service_context=service_context, show_progress=True
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return eval_results

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate an embedding model.")
    parser.add_argument("-m", "--modelname", type=str, required=True, help="Name of the embedding model")
    parser.add_argument("-d", "--directoryname", type=str, required=True, help="Name of the directory to search for files")
    args = parser.parse_args()

    # Search for files in the specified directory recursively
    file_paths = []
    for root, dirs, files in os.walk(args.directoryname):
        for file in files:
            file_paths.append(os.path.join(root, file))

    # Load your dataset using llama_index
    train_nodes = load_corpus(file_paths, verbose=True)

    # Generate synthetic queries using LLM (gpt-3.5-turbo)
    train_dataset = generate_qa_embedding_pairs(llm=OpenAI(model="gpt-3.5-turbo"), nodes=train_nodes)

    train_dataset.save_json("train_dataset.json")

    # Load the dataset
    train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")

    # Run Embedding Finetuning
    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,
        model_id=args.modelname,
        model_output_path="test_model",
    )
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()

    # Evaluate the Finetuned Model
    val_results_finetuned = evaluate(val_dataset, embed_model)
    df_finetuned = pd.DataFrame(val_results_finetuned)
    hit_rate_finetuned = df_finetuned["is_hit"].mean()
    print(f"Hit rate for the finetuned model: {hit_rate_finetuned}")

    # Alternatively, use InformationRetrievalEvaluator for a more comprehensive evaluation
    evaluate_st(val_dataset, args.modelname, name="finetuned")

if __name__ == "__main__":
    main()

