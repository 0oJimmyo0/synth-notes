from src.model import LlamaForEmbeddingLM
from src.utils import pairwise_cosine_similarity, batch_inference

import pickle
import argparse
from collections import defaultdict

import torch
import numpy as np

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description='ELM model evaluation tool')
    parser.add_argument('--backbone_model_path', type=str, default="initial_elm_model",
                        help='Path to the backbone model')
    parser.add_argument('--peft_model_id', type=str, default="5tasks_full_tuning_lora_outputs/checkpoint-37780",
                        help='Path to the PEFT model')
    parser.add_argument('--embedding_model_path', type=str, default="BAAI/bge-large-en-v1.5",
                    help='Path to the embedding model')
    parser.add_argument('--test_data_path', type=str, default="../data/pubmed_rct/processed_dataset/test_with_embeddings.pkl",
                        help='Path to the test data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help='Value for repetition penalty (use > 1.0 for abstract)')
    parser.add_argument('--num_eval_tasks', type=int, default=3,
                        help='Number of evaluation tasks to run (1-3)')
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

    # load testing data
    print(f"Loading test data from {args.test_data_path}")
    with open(args.test_data_path, 'rb') as f:
        test_dict = pickle.load(f)
    test_abstracts = test_dict["abstracts"]
    test_sections = test_dict["sections"] 
    test_section_texts = test_dict["section_texts"]
    test_summaries = test_dict["summaries"]
    test_embeddings = test_dict["embeddings"]

    # Task 1: Emb2Abstract
    if args.num_eval_tasks >= 1:
        print("\nRunning Task 1: Emb2Abstract")
        list_of_cos = []
        number_of_cases = len(test_abstracts)

        for i in range(0, number_of_cases, args.batch_size):
            testembs = test_embeddings[i:i+args.batch_size] # abstract embedding for input
            decoded_outputs = batch_inference(lora_elm, tokenizer, testembs, args.device, task="abstract", repetition_penalty=args.repetition_penalty)
            decoded_embs = embedding_model.encode(decoded_outputs)
            coss = pairwise_cosine_similarity(testembs, decoded_embs)
            list_of_cos += coss

        print(f"Abstract: {len(list_of_cos)} samples, Mean: {np.mean(list_of_cos):.4f}, Std: {np.std(list_of_cos):.4f}")

    # Task 2: Emb2Section
    if args.num_eval_tasks >= 2:
        print("\nRunning Task 2: Emb2Section")
        test_section_data_dict = defaultdict(lambda: defaultdict(list))
        for section, emb, text in zip(test_sections, test_embeddings, test_section_texts):
            test_section_data_dict[section.lower()]["embeddings"].append(emb)
            test_section_data_dict[section.lower()]["texts"].append(text)

        # evaluate each section
        for key in test_section_data_dict.keys():
            list_of_cos = []
            target_section = key
            number_of_cases = len(test_section_data_dict[target_section]["embeddings"])

            for i in range(0, number_of_cases, args.batch_size):
                testembs = test_section_data_dict[target_section]["embeddings"][i:i+args.batch_size]
                decoded_outputs = batch_inference(lora_elm, tokenizer, testembs, args.device, task=target_section)

                encoded_section_embs = embedding_model.encode(test_section_data_dict[target_section]["texts"][i:i+args.batch_size])
                decoded_embs = embedding_model.encode(decoded_outputs)
                coss = pairwise_cosine_similarity(encoded_section_embs, decoded_embs)
                list_of_cos += coss
                
            print(f"{target_section}: {len(list_of_cos)} samples, Mean: {np.mean(list_of_cos):.4f}, Std: {np.std(list_of_cos):.4f}")

    # Task 3: Emb2PLS
    if args.num_eval_tasks >= 3:
        print("\nRunning Task 3: Emb2PLS")
        list_of_cos = []
        number_of_cases = len(test_summaries)

        for i in range(0, number_of_cases, args.batch_size):
            testembs = test_embeddings[i:i+args.batch_size] # abstract embedding for input
            decoded_outputs = batch_inference(lora_elm, tokenizer, testembs, args.device, task="pls")

            decoded_embs = embedding_model.encode(decoded_outputs)
            coss = pairwise_cosine_similarity(testembs, decoded_embs)
            list_of_cos += coss

        print(f"PLS: {len(list_of_cos)} samples, Mean: {np.mean(list_of_cos):.4f}, Std: {np.std(list_of_cos):.4f}")

if __name__ == "__main__":
    main()