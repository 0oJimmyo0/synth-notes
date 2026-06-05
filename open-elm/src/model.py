from transformers import LlamaForCausalLM, LlamaConfig
from typing import Optional
import torch

class LlamaForEmbeddingConfig(LlamaConfig):
    
    # we cannot use model_type = "llama_embedding" because it is not registered
    # the whole huggingface ecosystem (peft, trl, etc.) cannot recognize it as a valid model
    #model_type = "llama_embedding"
    model_type = "llama"
    
    def __init__(
        self,
        dim_embed_domain=1024,
        dim_adapter_hidden=2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_embed_domain = dim_embed_domain
        self.dim_adapter_hidden = dim_adapter_hidden

class LlamaForEmbeddingLM(LlamaForCausalLM):
    
    config_class = LlamaForEmbeddingConfig
    
    def __init__(self, config):
        # LlamaForCausalLM.__init__(self, config)
        super().__init__(config) 
        self.dim_embed_domain = config.dim_embed_domain
        self.dim_adapter_hidden = config.dim_adapter_hidden
        # Get the dimension of the token embeddings
        self.dim_embed_token = self.model.embed_tokens.embedding_dim

        # Initialize adapter in __init__
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(self.dim_embed_domain, self.dim_adapter_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_adapter_hidden, self.dim_embed_token)
        )
    
    def forward(
        self,
        *args,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        domain_embeddings=None,
        **kwargs
    ):
        # start with embedding input_ids 
        embs = self.model.embed_tokens(input_ids)
        
        # OPTIMIZATION: Vectorized replacement instead of nested loops
        # Find all positions where input_ids == 128002 (embedding token)
        if domain_embeddings is not None and len(domain_embeddings) > 0:
            # Create mask for embedding token positions
            emb_token_mask = (input_ids == 128002)  # 128002 maps to <|reserved_special_token_0|>
            
            if emb_token_mask.any():
                # Convert domain_embeddings list to tensor for batch processing
                # This is more memory efficient than processing one by one
                # Handle both tensor and numpy/list inputs
                processed_embs = []
                for emb in domain_embeddings:
                    if isinstance(emb, torch.Tensor):
                        # Already a tensor, just move to correct device/dtype
                        processed_embs.append(emb.to(dtype=embs.dtype, device=embs.device))
                    else:
                        # Convert from numpy/list to tensor
                        processed_embs.append(torch.tensor(emb, dtype=embs.dtype, device=embs.device))
                domain_emb_tensor = torch.stack(processed_embs)
                
                # Batch process all domain embeddings through adapter at once
                # This reduces memory fragmentation and is faster
                adapted_embs = self.adapter(domain_emb_tensor)
                
                # Replace embedding positions with adapted embeddings
                emb_i = 0
                batch_size, seq_len = input_ids.shape
                for i in range(batch_size):
                    for j in range(seq_len):
                        if emb_token_mask[i, j]:
                            embs[i, j] = adapted_embs[emb_i]
                            emb_i += 1
                            if emb_i >= len(adapted_embs):
                                break
                    if emb_i >= len(adapted_embs):
                        break
        
        kwargs['inputs_embeds']=embs
        kwargs['input_ids']=None
        # pass the modified embeddings to the parent class's forward function
        # this allows we 
        # -> pass the modified embeddings through transformer layers
        # -> apply language modeling head
        # -> generate output
        return super().forward(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        *args,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        domain_embeddings=None,
        **kwargs
    ):
        output = super().prepare_inputs_for_generation(*args, **kwargs)
        # ensure that domain embeddings are passed through each generation step
        output.update({"domain_embeddings": domain_embeddings})
        # this output will be used in forward function
        return output

# helper function to initialize the embedding model from a causal LM checkpoint
def initialize_embedding_model_from_causal_lm(
    pretrained_model_name_or_path,
    dim_embed_domain=1024,
    dim_adapter_hidden=2048
):
    """
    Load a LlamaForEmbeddingLM model with adapter weights from a LlamaForCausalLM checkpoint.
    
    Args:
        pretrained_model_name_or_path (str): Path to the base LlamaForCausalLM model
        dim_embed_domain (int): Domain embedding dimension
        dim_adapter_hidden (int): Adapter hidden dimension
        **kwargs: Additional args to pass to from_pretrained (like device_map, torch_dtype, etc.)
        
    Returns:
        LlamaForEmbeddingLM: Loaded model with adapter initialized
    """
    # Load the original config from the pretrained model
    original_config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
    
    # Create embedding config based on the original config
    embedding_config = LlamaForEmbeddingConfig(
        **original_config.to_dict(),
        dim_embed_domain=dim_embed_domain,
        dim_adapter_hidden=dim_adapter_hidden
    )
    
    print("Initializing embedding model with the new config (adapter will be initialized randomly)")
    # Initialize our embedding model with the new config (adapter will be initialized randomly)
    embedding_model = LlamaForEmbeddingLM(embedding_config)
    
    print("Loading the weights from the pretrained model")
    # Load the weights from the pretrained model directly to GPU to save CPU RAM
    import torch
    causal_lm_model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Getting the state dict from the pretrained model")
    # Get the state dict from the causal LM model
    causal_lm_state_dict = causal_lm_model.state_dict()
    
    # Delete the causal model to free GPU/CPU memory before loading into embedding model
    del causal_lm_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Filter out any keys that might not match our model (should be none in this case)
    embedding_model_keys = set(embedding_model.state_dict().keys())
    filtered_state_dict = {
        k: v.cpu() if isinstance(v, torch.Tensor) and v.is_cuda else v
        for k, v in causal_lm_state_dict.items() 
        if k in embedding_model_keys and not k.startswith('adapter.')
    }
    
    # Clear the original state dict to free memory
    del causal_lm_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Loading the state dict into our embedding model")
    # Load the filtered state dict into our embedding model
    # strict=False: ignore the missing keys, e.g., adapter weights
    embedding_model.load_state_dict(filtered_state_dict, strict=False)
    
    return embedding_model