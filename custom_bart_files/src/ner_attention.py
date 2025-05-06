import torch

def create_ner_bias_mask(ner_mask: torch.Tensor, ner_bias: float = -1.0):
    """
    Create an additive attention bias mask for NER tokens.
    Returns a mask tensor of shape [batch, 1, 1, seq_len] with 0 for entity tokens and `ner_bias` for non-entities.
    """
    # ner_mask: 1 for entity, 0 for not entity
    if ner_mask.dim() != 2:
        raise ValueError("ner_mask should be 2D (batch, seq_len)")
    # 1 for non-entity positions
    non_entity_mask = (ner_mask == 0).to(torch.float32)
    # Expand to [batch, 1, 1, seq_len]
    bias_mask = non_entity_mask[:, None, None, :] * ner_bias
    return bias_mask

def apply_ner_attention_bias(attn_scores: torch.Tensor, ner_mask: torch.Tensor, ner_bias: float = -0.05):
    """
    Apply NER bias to given attention scores (additive).
    """
    bias_mask = create_ner_bias_mask(ner_mask.to(attn_scores.device), ner_bias)
    # Ensure bias_mask shape matches attn_scores [..., seq_len]
    if bias_mask.shape[-1] != attn_scores.shape[-1]:
        raise ValueError("NER bias mask length does not match attention scores length.")
    return attn_scores + bias_mask
