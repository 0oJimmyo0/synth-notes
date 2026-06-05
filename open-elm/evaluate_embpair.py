from src.model import LlamaForEmbeddingLM
from src.utils import pairwise_cosine_similarity, batch_pair_inference

import pickle
import argparse

import torch
import numpy as np

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description='ELM model evaluation tool for doc pair inference')
    parser.add_argument('--backbone_model_path', type=str, default="initial_elm_model",
                        help='Path to the backbone model')
    parser.add_argument('--peft_model_id', type=str, default="5tasks_full_tuning_lora_outputs/checkpoint-37780",
                        help='Path to the PEFT model')
    parser.add_argument('--embedding_model_path', type=str, default="BAAI/bge-large-en-v1.5",
                    help='Path to the embedding model')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
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

    def run_and_report(embs_i, embs_j, analyses, task):

        list_of_cos = []
        number_of_cases = len(analyses)

        for i in range(0, number_of_cases, args.batch_size):
            testembs_i = embs_i[i:i+args.batch_size]
            testembs_j = embs_j[i:i+args.batch_size]
            decoded_outputs = batch_pair_inference(lora_elm, tokenizer, testembs_i, testembs_j, args.device, task)
            reference_embs = embedding_model.encode(analyses[i:i+args.batch_size])
            decoded_embs = embedding_model.encode(decoded_outputs)
            coss = pairwise_cosine_similarity(reference_embs, decoded_embs)
            list_of_cos += coss

        print(f"{task}: {len(list_of_cos)} samples, Mean: {np.mean(list_of_cos):.4f}, Std: {np.std(list_of_cos):.4f}")
    
    # load testing data
    print(f"Loading pairs within same topic for commonality analysis")
    with open(f'../data/pubmed_rct/processed_dataset/test_within_same_topic_commonalities_with_embeddings.pkl', 'rb') as pkl:
        test_within_same_topic_commonalities_dict = pickle.load(pkl)
    test_within_same_topic_commonalities_embs_i = test_within_same_topic_commonalities_dict["embs_i"]
    test_within_same_topic_commonalities_embs_j = test_within_same_topic_commonalities_dict["embs_j"]
    test_within_same_topic_commonalities_analyses = test_within_same_topic_commonalities_dict["analyses"]
    run_and_report(test_within_same_topic_commonalities_embs_i, test_within_same_topic_commonalities_embs_j, 
                   test_within_same_topic_commonalities_analyses, "commonality")

    print(f"Loading pairs across different topics for commonality analysis")
    with open(f'../data/pubmed_rct/processed_dataset/test_across_different_topics_commonalities_with_embeddings.pkl', 'rb') as pkl:
        test_across_different_topics_commonalities_dict = pickle.load(pkl)
    test_across_different_topics_commonalities_embs_i = test_across_different_topics_commonalities_dict["embs_i"]
    test_across_different_topics_commonalities_embs_j = test_across_different_topics_commonalities_dict["embs_j"]
    test_across_different_topics_commonalities_analyses = test_across_different_topics_commonalities_dict["analyses"]
    run_and_report(test_across_different_topics_commonalities_embs_i, test_across_different_topics_commonalities_embs_j, 
                   test_across_different_topics_commonalities_analyses, "commonality")

    print(f"Loading pairs within same topic for difference analysis")
    with open(f'../data/pubmed_rct/processed_dataset/test_within_same_topic_differences_with_embeddings.pkl', 'rb') as pkl:
        test_within_same_topic_differences_dict = pickle.load(pkl)
    test_within_same_topic_differences_embs_i = test_within_same_topic_differences_dict["embs_i"]
    test_within_same_topic_differences_embs_j = test_within_same_topic_differences_dict["embs_j"]
    test_within_same_topic_differences_analyses = test_within_same_topic_differences_dict["analyses"]
    run_and_report(test_within_same_topic_differences_embs_i, 
                   test_within_same_topic_differences_embs_j, 
                   test_within_same_topic_differences_analyses, "difference")

    print(f"Loading pairs across different topics for difference analysis")
    with open(f'../data/pubmed_rct/processed_dataset/test_across_different_topics_differences_with_embeddings.pkl', 'rb') as pkl:
        test_across_different_topics_differences_dict = pickle.load(pkl)
    test_across_different_topics_differences_embs_i = test_across_different_topics_differences_dict["embs_i"]
    test_across_different_topics_differences_embs_j = test_across_different_topics_differences_dict["embs_j"]
    test_across_different_topics_differences_analyses = test_across_different_topics_differences_dict["analyses"]
    run_and_report(test_across_different_topics_differences_embs_i, 
                   test_across_different_topics_differences_embs_j, 
                   test_across_different_topics_differences_analyses, "difference")

if __name__ == "__main__":
    main()