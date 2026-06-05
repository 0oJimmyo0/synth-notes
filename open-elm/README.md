# Embedding Language Model (ELM)

**Project design:** For a structured design plan (data → embeddings → ELM → downstream mortality) for mentor assessment and potential publication, see [DESIGN_PLAN.md](../DESIGN_PLAN.md) in the parent directory.

Please use Python 3.11 and `pip intsall -r requirements.txt` to set up the necessary dependencies.

# Model Class & Utilities
- `src/model.py` defines config and model class 
- `src/utils.py` defines helper functions, e.g., collate and inference

# Training
- `initialize_model.ipynb` creates initialized elm based on llama model
- `p1_train.py` fine-tunes only adapter in the elm
- `train.py` fine-tunes both adapter and transformer modules (using LoRA) in the elm

# Evaluation

- `evaluate.py` loads fine-tuned elm and performs inference on embedding2abstract, embedding2section, and embedding2pls. All tasks take one embedding as the input.
- `evaluate_embpair.py` loads fine-tuned elm and performs inference on embedding2commonality and embedding2difference. Both tasks take a pair of embeddings as the input.
- `evaluate_interpolation.py` is similar to `evaluate.py` but takes interpolated embedding as input and performs only embedding2abstract and embedding2pls.
    - `generate_interpolated_embeddings.ipynb` generates interpolated embedding.

## Concept Activation Vector (CAV)
- `{gender|age}_cav.ipynb` load CAV data, fit regression model to find either gender or age concept vector, and load fine-tuned elm to generate CAV-guided abstracts (embedding2abstract).
- `{gender|age}_cav_analysis.ipynb` evaluate CAV-guided abstracts by extracting gender or age to see if it changes along with the CAV. 

## Win Rate 
- `winrate_analysis.ipynb` ask LLM to tell plausible abstracts between real abstracts and elm-generated abstracts.