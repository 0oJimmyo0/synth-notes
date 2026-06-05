import torch
import numpy as np

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate_function(examples):
    input_ids = []
    labels = []
    for example in examples:
        # Find the position of the generation token
        gen_tok_pos = example["input_ids"].index(128003) # 128003 maps to <|reserved_special_token_1|>, which we define it as generation token 
        # Create input_ids without the generation token
        input_ids.append(torch.tensor(example["input_ids"][:gen_tok_pos] + example["input_ids"][gen_tok_pos+1:], dtype=torch.long))
        # Create labels with -100 (ignored in loss calculation) for tokens before generation token
        # and actual tokens after generation token
        labels.append(torch.tensor([-100]*gen_tok_pos + example["input_ids"][gen_tok_pos+1:], dtype=torch.long))
    
    embs = [torch.tensor(x) for example in examples for x in example["domain_embeddings"]]
    
    return {"input_ids": torch.stack(input_ids), "domain_embeddings": embs, "labels": torch.stack(labels)}

def collate_function_dynamic_padding(examples, max_seq_length=None):
    """
    Collate function with dynamic padding that respects max_seq_length.
    
    Args:
        examples: List of examples to collate
        max_seq_length: Maximum sequence length to enforce (truncates if longer)
    """
    input_ids = []
    labels = []
    # -1 because we will not include generation token (i.e., 128003) in the resultant input_ids and labels
    max_length_in_batch = max([len(example["input_ids"]) for example in examples]) - 1
    
    # CRITICAL FIX: Enforce max_seq_length to prevent OOM
    # If max_seq_length is provided, use the minimum of batch max and max_seq_length
    if max_seq_length is not None:
        max_length = min(max_length_in_batch, max_seq_length)
    else:
        max_length = max_length_in_batch
    
    # use <|eot_id|> (i.e., 128009) for padding
    pad_token_id = 128009 

    for example in examples:
        gen_tok_pos = example["input_ids"].index(128003)
        # filter out generation token
        ids_without_gen_token = example["input_ids"][:gen_tok_pos] + example["input_ids"][gen_tok_pos+1:]
        
        # CRITICAL FIX: Truncate if sequence exceeds max_seq_length
        if max_seq_length is not None and len(ids_without_gen_token) > max_seq_length:
            # Truncate from the end, preserving the prompt (before gen_tok_pos)
            # Keep as much of the target as possible
            prompt_length = gen_tok_pos
            target_length = len(ids_without_gen_token) - prompt_length
            available_target_length = max_seq_length - prompt_length
            
            if available_target_length > 0:
                # Truncate target to fit
                ids_without_gen_token = ids_without_gen_token[:prompt_length + available_target_length]
            else:
                # If prompt itself is too long, truncate it
                ids_without_gen_token = ids_without_gen_token[:max_seq_length]
                gen_tok_pos = min(gen_tok_pos, max_seq_length)
        
        # create an array with max_length, filled up by pad_token_id
        input_ids_padded = torch.full((max_length,), pad_token_id, dtype=torch.long)
        actual_length = min(len(ids_without_gen_token), max_length)
        input_ids_padded[:actual_length] = torch.tensor(ids_without_gen_token[:actual_length], dtype=torch.long)
        input_ids.append(input_ids_padded)

        labels_padded = torch.full((max_length,), -100, dtype=torch.long)
        # set prompt [:gen_tok_pos] as -100
        # set pads [actual_length:] as -100
        # only learn target, which is [gen_tok_pos:actual_length]
        target_start = min(gen_tok_pos, actual_length)
        target_end = min(actual_length, len(ids_without_gen_token))
        if target_start < target_end:
            labels_padded[target_start:target_end] = input_ids_padded[target_start:target_end]
        labels.append(labels_padded)

    embs = [torch.tensor(x) for example in examples for x in example["domain_embeddings"]] 

    return {"input_ids": torch.stack(input_ids), "domain_embeddings": embs, "labels": torch.stack(labels)}

def pairwise_cosine_similarity(a, b):
    # Compute dot products row-wise
    dot_products = np.sum(a * b, axis=1)
    
    # Compute norms
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    
    # Avoid division by zero
    norm_product = norm_a * norm_b
    norm_product[norm_product == 0] = 1e-8  # small epsilon to avoid zero-division
    
    # Compute cosine similarity
    cosine_sim = dot_products / norm_product
    return cosine_sim.tolist()

def batch_inference(model, tokenizer, embeddings, device, task="abstract", repetition_penalty=1.0,
                    temperature=None, top_p=None, top_k=None, max_new_tokens=1024, do_sample=False):
    """
    Run inference in batch mode.
    
    Args:
        model: The model to use for inference
        tokenizer: The tokenizer for the model
        embeddings: List of embeddings for each input
        device: Device to run inference on (cuda or cpu)
        task: The task to run inference for
        repetition_penalty: Penalty for repetition (default: 1.0)
        temperature: Sampling temperature (None = greedy, higher = more creative)
        top_p: Nucleus sampling parameter (None = disabled)
        top_k: Top-k sampling parameter (None = disabled)
        max_new_tokens: Maximum number of new tokens to generate (default: 1024)
        do_sample: Whether to use sampling (default: False, uses greedy if False)
                
    Returns:
        List of generated outputs
    """
    
    # Process in batches
    input_ids_list = []
    for i in range(0, len(embeddings)):
        
        # Prepare inputs for each item in the batch
        if task == "abstract":
            chat = [
                {"role": "user", "content": "Provide the text of the abstract <|reserved_special_token_0|>"},
            ]
        elif task == "background":
            chat = [
                {"role": "user", "content": "Write the background section for the abstract <|reserved_special_token_0|>"},
            ]
        elif task == "objective":
            chat = [
                {"role": "user", "content": "Write the objective section for the abstract <|reserved_special_token_0|>"},
            ]
        elif task == "methods":
            chat = [
                {"role": "user", "content": "Write the methods section for the abstract <|reserved_special_token_0|>"},
            ]
        elif task == "results":
            chat = [
                {"role": "user", "content": "Write the results section for the abstract <|reserved_special_token_0|>"},
            ]
        elif task == "conclusions":
            chat = [
                {"role": "user", "content": "Write the conclusions section for the abstract <|reserved_special_token_0|>"},
            ]
        elif task == "pls":
            chat = [
                {"role": "user", "content": "Please write a plain language summary of the abstract <|reserved_special_token_0|>"},
            ]
        elif task == "clinic_note":
            chat = [
                {"role": "user", "content": "Provide the text of the clinic note <|reserved_special_token_0|>"},
            ]
            
        input_ids = tokenizer.apply_chat_template(chat, return_tensors='pt', add_generation_prompt=True).to(device)
        input_ids_list.append(input_ids)
        
    # Get max length for padding
    max_length = max(ids.shape[1] for ids in input_ids_list)
        
    # Pad all inputs to max length and create attention masks
    padded_inputs = []
    attention_masks = []
    prompt_lengths = []
    for input_ids in input_ids_list:
        prompt_length = input_ids.shape[1]
        prompt_lengths.append(prompt_length)
        
        # Pad to max length
        padding_length = max_length - prompt_length
        if padding_length > 0:
            padding = torch.full((1, padding_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
            padded_input = torch.cat([input_ids, padding], dim=1)
            # Create attention mask: 1 for real tokens, 0 for padding
            attention_mask = torch.cat([
                torch.ones((1, prompt_length), dtype=torch.long, device=device),
                torch.zeros((1, padding_length), dtype=torch.long, device=device)
            ], dim=1)
        else:
            padded_input = input_ids
            attention_mask = torch.ones((1, prompt_length), dtype=torch.long, device=device)
        
        padded_inputs.append(padded_input)
        attention_masks.append(attention_mask)
    
    # Stack all inputs into a batch
    batch_input_ids = torch.cat(padded_inputs, dim=0)
    batch_attention_mask = torch.cat(attention_masks, dim=0)
    
    # Convert embeddings to tensor
    batch_embs_tensor = [torch.tensor(emb, dtype=torch.bfloat16).to(device) for emb in embeddings]
    
    # Prepare generation kwargs
    # Use max_new_tokens instead of max_length to specify tokens to generate beyond the prompt
    generation_kwargs = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,  # Explicitly tell model which tokens to attend to
        "domain_embeddings": batch_embs_tensor,
        "max_new_tokens": max_new_tokens,  # Number of new tokens to generate (not including prompt)
        "eos_token_id": [128009],  # eot_id: 128009
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }
    
    # Add sampling parameters if provided
    if do_sample or temperature is not None or top_p is not None or top_k is not None:
        generation_kwargs["do_sample"] = True
        if temperature is not None:
            generation_kwargs["temperature"] = temperature
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if top_k is not None:
            generation_kwargs["top_k"] = top_k
    else:
        generation_kwargs["do_sample"] = False
    
    # Generate outputs
    outputs = model.generate(**generation_kwargs)
    
    # Decode outputs
    results = []
    for j, output in enumerate(outputs):
        prompt_length = prompt_lengths[j]
        result = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
        results.append(result)
    
    return results

def batch_pair_inference(model, tokenizer, embeddings_i, embeddings_j, device, task="commonality"):
    """
    Run pair inference in batch mode.
    
    Args:
        model: The model to use for inference
        tokenizer: The tokenizer for the model
        embeddings_i: List of first embeddings for each pair
        embeddings_j: List of second embeddings for each pair
        device: Device to run inference on (cuda or cpu)
        task: Type of comparison task (commonality or difference)
                
    Returns:
        List of generated outputs
    """
    assert len(embeddings_i) == len(embeddings_j), "Number of first embeddings must match number of second embeddings"
    
    # Prepare inputs for each item in the batch
    input_ids_list = []
    for i in range(len(embeddings_i)):
        if task == "commonality":
            chat = [
                {"role": "user", "content": "List five commonalities between the first abstract <|reserved_special_token_0|> and the second abstract <|reserved_special_token_0|>"},
            ]
        elif task == "difference":
            chat = [
                {"role": "user", "content": "List five differences between the first abstract <|reserved_special_token_0|> and the second abstract <|reserved_special_token_0|>"},
            ]
            
        input_ids = tokenizer.apply_chat_template(chat, return_tensors='pt', add_generation_prompt=True).to(device)
        input_ids_list.append(input_ids)
    
    # Get max length for padding
    max_length = max(ids.shape[1] for ids in input_ids_list)
    
    # Pad all inputs to max length and create attention masks
    padded_inputs = []
    attention_masks = []
    prompt_lengths = []
    for input_ids in input_ids_list:
        prompt_length = input_ids.shape[1]
        prompt_lengths.append(prompt_length)
        
        # Pad to max length
        padding_length = max_length - prompt_length
        if padding_length > 0:
            padding = torch.full((1, padding_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
            padded_input = torch.cat([input_ids, padding], dim=1)
            # Create attention mask: 1 for real tokens, 0 for padding
            attention_mask = torch.cat([
                torch.ones((1, prompt_length), dtype=torch.long, device=device),
                torch.zeros((1, padding_length), dtype=torch.long, device=device)
            ], dim=1)
        else:
            padded_input = input_ids
            attention_mask = torch.ones((1, prompt_length), dtype=torch.long, device=device)
        
        padded_inputs.append(padded_input)
        attention_masks.append(attention_mask)
    
    # Stack all inputs into a batch
    batch_input_ids = torch.cat(padded_inputs, dim=0)
    batch_attention_mask = torch.cat(attention_masks, dim=0)
    
    # Convert embeddings to tensor pairs
    batch_embs_pairs = []
    for i in range(len(embeddings_i)):
        emb_i_tensor = torch.tensor(embeddings_i[i], dtype=torch.bfloat16).to(device)
        emb_j_tensor = torch.tensor(embeddings_j[i], dtype=torch.bfloat16).to(device)
        batch_embs_pairs.append(emb_i_tensor)
        batch_embs_pairs.append(emb_j_tensor)
    
    # Generate outputs
    outputs = model.generate(
        input_ids=batch_input_ids,
        domain_embeddings=batch_embs_pairs,
        max_length=512,
        eos_token_id=[128009],  # eot_id: 128009
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode outputs
    results = []
    for j, output in enumerate(outputs):
        prompt_length = prompt_lengths[j]
        result = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
        results.append(result)
    
    return results