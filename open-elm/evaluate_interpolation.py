from src.model import LlamaForEmbeddingLM
from src.utils import pairwise_cosine_similarity, batch_inference

import pickle
import argparse

import torch
import numpy as np

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description='ELM model evaluation tool for interpolated embeddings')
    parser.add_argument('--backbone_model_path', type=str, default="initial_elm_model",
                        help='Path to the backbone model')
    parser.add_argument('--peft_model_id', type=str, default="5tasks_full_tuning_lora_outputs/checkpoint-37780",
                        help='Path to the PEFT model')
    parser.add_argument('--embedding_model_path', type=str, default="BAAI/bge-large-en-v1.5",
                    help='Path to the embedding model')
    parser.add_argument('--interpolated_data_path', type=str, default="../data/pubmed_rct/processed_dataset/interpolated_embeddings.pkl",
                        help='Path to the interpolated data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help='Value for repetition penalty (use > 1.0 for abstract)')
    parser.add_argument('--eval_task', type=str, default="both",
                        help='evaluate model on abstract, pls, or both')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # load embedding model
    print(f"Loading embedding model from {args.embedding_model_path}")
    embedding_model = SentenceTransformer(args.embedding_model_path)

    # load tokenizer & elm
    print(f"Loading backbone model from {args.backbone_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)

    elm = LlamaForEmbeddingLM.from_pretrained(
        args.backbone_model_path, 
        torch_dtype=torch.bfloat16,
        device_map=args.device)

    print(f"Loading PEFT model from {args.peft_model_id}")
    lora_elm = PeftModel.from_pretrained(elm, args.peft_model_id)
    lora_elm = lora_elm.merge_and_unload()

    # load interpolated data
    print(f"Loading interpolated data from {args.interpolated_data_path}")
    with open(args.interpolated_data_path, 'rb') as f:
        interpolated_dict = pickle.load(f)
    interpolated_embeddings = interpolated_dict["embeddings"]

    # Emb2Abstract
    if args.eval_task == "abstract" or args.eval_task == "both":
        print("\nRunning Emb2Abstract")
        list_of_cos = []
        number_of_cases = len(interpolated_embeddings)

        for i in range(0, number_of_cases, args.batch_size):
            testembs = interpolated_embeddings[i:i+args.batch_size] # abstract embedding for input
            decoded_outputs = batch_inference(lora_elm, tokenizer, testembs, args.device, task="abstract", repetition_penalty=args.repetition_penalty)
            decoded_embs = embedding_model.encode(decoded_outputs)
            coss = pairwise_cosine_similarity(testembs, decoded_embs)
            list_of_cos += coss

        print(f"Abstract: {len(list_of_cos)} samples, Mean: {np.mean(list_of_cos):.4f}, Std: {np.std(list_of_cos):.4f}")

    
    # Emb2PLS
    if args.eval_task == "pls" or args.eval_task == "both":
        print("\nRunning Emb2PLS")
        list_of_cos = []
        number_of_cases = len(interpolated_embeddings)

        for i in range(0, number_of_cases, args.batch_size):
            testembs = interpolated_embeddings[i:i+args.batch_size] # abstract embedding for input
            decoded_outputs = batch_inference(lora_elm, tokenizer, testembs, args.device, task="pls")
            decoded_embs = embedding_model.encode(decoded_outputs)
            coss = pairwise_cosine_similarity(testembs, decoded_embs)
            list_of_cos += coss

        print(f"PLS: {len(list_of_cos)} samples, Mean: {np.mean(list_of_cos):.4f}, Std: {np.std(list_of_cos):.4f}")

if __name__ == "__main__":
    main()