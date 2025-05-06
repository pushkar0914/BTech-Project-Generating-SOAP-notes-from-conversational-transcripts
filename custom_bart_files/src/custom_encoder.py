import math
import torch
import torch.nn as nn
from transformers import BartConfig

class CustomBartEncoder(nn.Module):
    """
    Custom BART Encoder with NER-enhanced self-attention and section/subsection embeddings.
    - Injects section and subsection embeddings into the token embeddings.
    - Applies a bias in self-attention to emphasize NER tokens (ner_mask).
    - Handles long inputs by chunking and attention pooling if seq_len > max_source_positions.
    """
    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.config = config
        #print(f"encoder config :{config}")
        self.embed_tokens = embed_tokens  # shared token embeddings
        #print(f"[encoder] tokens :{embed_tokens}")
        embed_dim = config.d_model
        #print(f"[encoder] embed_dim :{embed_dim}")
        self.padding_idx = config.pad_token_id
        #print(f"[encoder] padding_idx :{self.padding_idx}")
        self.max_source_positions = config.max_position_embeddings
        #print(f"[encoder] max src posn :{self.max_source_positions}")


        # Learned positional embeddings (Bart uses offset=2 hack for positional ids)
        self.embed_positions = nn.Embedding(config.max_position_embeddings + 2, embed_dim)
        nn.init.normal_(self.embed_positions.weight, mean=0, std=0.02)  # initialize like BART
        self.embed_positions.weight.data[self.padding_idx] = 0  # zero out pad position embedding

        # Section and subsection embeddings
        self.section_embed = nn.Embedding(4, embed_dim)
        self.subsection_embed = nn.Embedding(15, embed_dim)
        nn.init.normal_(self.section_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.subsection_embed.weight, mean=0, std=0.02)

        # Layernorm for embedding outputs
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.dropout = config.dropout

        # Encoder layers
        self.layers = nn.ModuleList([CustomBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        # (CustomBartEncoderLayer is defined below, inside this file or imported)

    def forward(self, input_ids=None, attention_mask=None, section_ids=None, subsection_ids=None, ner_mask=None, inputs_embeds=None):
        """
        Forward pass for the encoder.
        Args:
            input_ids (torch.LongTensor): [batch, seq_len] input token IDs.
            attention_mask (torch.Tensor): [batch, seq_len] attention mask (1 for tokens, 0 for padding).
            section_ids (torch.LongTensor): [batch, seq_len] section labels per token (0-3 for S,O,A,P).
            subsection_ids (torch.LongTensor): [batch, seq_len] subsection labels per token (0-14).
            ner_mask (torch.Tensor): [batch, seq_len] binary mask indicating NER tokens (1=entity, 0=non-entity).
            inputs_embeds (torch.FloatTensor): Optional precomputed embeddings [batch, seq_len, embed_dim].
        Returns:
            torch.FloatTensor of shape [batch, enc_seq_len, embed_dim]: Encoder output states.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide input_ids or inputs_embeds to the encoder.")

        # Get shapes
        if input_ids is not None:
           # print(f"input_ids_encoder :{input_ids}")
            batch_size, seq_len = input_ids.shape
            #print(f"[encoder] batch_size :{batch_size},seq len : {seq_len}")

        else:
            batch_size, seq_len, _ = inputs_embeds.shape

        # Debug: initial shape
        #print(f"[Encoder] input shape: { (batch_size, seq_len) }")

        # If input too long, apply chunking
        if seq_len > self.max_source_positions:
            # Calculate number of chunks
            chunk_size = self.max_source_positions
            chunks = []
            # Split inputs into chunks
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                if input_ids is not None:
                    chunk_ids = input_ids[:, start:end]
                    chunk_embeds = self.embed_tokens(chunk_ids)
                else:
                    chunk_embeds = inputs_embeds[:, start:end, :]
                # Compute embeddings with positions etc. for this chunk
                # We reset positional embedding for each chunk (no carry-over beyond max length)
                positions = (torch.arange(chunk_embeds.size(1), device=chunk_embeds.device)
                             .unsqueeze(0).expand(batch_size, -1))
                pos_embeds = self.embed_positions(positions)
                # Section/subsection for this chunk if provided
                if section_ids is not None:
                    sec_ids_chunk = section_ids[:, start:end]
                    if sec_ids_chunk.shape != chunk_embeds.shape[:2]:
                        raise ValueError("section_ids shape must match input_ids shape")
                    sec_embeds = self.section_embed(sec_ids_chunk)
                else:
                    sec_embeds = torch.zeros_like(chunk_embeds)
                if subsection_ids is not None:
                    subsec_ids_chunk = subsection_ids[:, start:end]
                    if subsec_ids_chunk.shape != chunk_embeds.shape[:2]:
                        raise ValueError("subsection_ids shape must match input_ids shape")
                    subsec_embeds = self.subsection_embed(subsec_ids_chunk)
                else:
                    subsec_embeds = torch.zeros_like(chunk_embeds)
                # Sum token + positional + section + subsection embeddings
                embeds = chunk_embeds + pos_embeds + sec_embeds + subsec_embeds
                embeds = self.layernorm_embedding(embeds)
                embeds = nn.functional.dropout(embeds, p=self.dropout, training=self.training)
                # Prepare attention mask for this chunk
                if attention_mask is not None:
                    chunk_mask = attention_mask[:, start:end]
                else:
                    # default attention mask if not provided: 1 for all tokens
                    chunk_mask = torch.ones((batch_size, chunk_embeds.size(1)), device=chunk_embeds.device)
                # Encode this chunk with encoder layers
                enc_states = embeds
                enc_attn_mask = self._prepare_encoder_attention_mask(chunk_mask, batch_size, chunk_embeds.size(1))
                for i, layer in enumerate(self.layers):
                    enc_states = layer(enc_states, enc_attn_mask, ner_mask=None)  # we do not use ner_mask per-chunk (optional)
                    # layer returns hidden_states (no attn weights)
                    if isinstance(enc_states, tuple):
                        enc_states = enc_states[0]
                chunks.append(enc_states)
            # Attention pooling: fuse chunk outputs
            # Here we use simple average pooling over chunks as a summary (more advanced can be applied).
            # Compute average of each chunk's output states over the time dimension
            chunk_reps = [state.mean(dim=1) for state in chunks]  # each is [batch, embed_dim]
            # Stack chunk representations
            fused_seq = torch.stack(chunk_reps, dim=1)  # [batch, num_chunks, embed_dim]
            # Optionally, one could apply an additional attention layer over chunk representations here.
            encoder_output = fused_seq
            # Debug:
            #print(f"[Encoder] input was long: split into {len(chunks)} chunks, fused output shape: {encoder_output.shape}")
            return encoder_output

        # If not chunking (seq_len <= max_source_positions):
        # Compute input embeddings
        if inputs_embeds is None:
            #print("input embed none")
            inputs_embeds = self.embed_tokens(input_ids)
            #print(f"input emned calculated !!")
        # Apply scaling if configured
        embed_scale = math.sqrt(self.config.d_model) if getattr(self.config, "scale_embedding", False) else 1.0
        inputs_embeds = inputs_embeds * embed_scale

        # Add positional embeddings
        positions = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, seq_len)
        # BartLearnedPositionalEmbedding uses offset; simulate by adding offset=2
        pos_embeds = self.embed_positions(positions + 2)
        # Section and subsection embeddings (optional)
        if section_ids is not None:
            if section_ids.shape != (batch_size, seq_len):
                raise ValueError("section_ids must have shape [batch_size, seq_len]")
            sec_embeds = self.section_embed(section_ids)
        else:
            sec_embeds = torch.zeros_like(inputs_embeds)
        if subsection_ids is not None:
            if subsection_ids.shape != (batch_size, seq_len):
                raise ValueError("subsection_ids must have shape [batch_size, seq_len]")
            subsec_embeds = self.subsection_embed(subsection_ids)
        else:
            subsec_embeds = torch.zeros_like(inputs_embeds)

        # Combine token, positional, section, subsection embeddings
        hidden_states = inputs_embeds + pos_embeds + sec_embeds + subsec_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Prepare attention mask (now returns a 3D mask)
        if attention_mask is None:
            # Default: no padding mask (all ones)
            attention_mask = torch.ones((batch_size, seq_len), device=hidden_states.device)
        enc_attn_mask = self._prepare_encoder_attention_mask(attention_mask, batch_size, seq_len)
        # Incorporate NER mask into attention (bias towards entity tokens)
        if ner_mask is not None:
            if ner_mask.shape != (batch_size, seq_len):
                raise ValueError("ner_mask must have shape [batch_size, seq_len]")
            ner_bias = -0.05 # a small negative bias for non-entities
            # Create a ner_bias_mask from ner_mask of shape [batch, src_len]
            ner_bias_mask = (ner_mask == 0).to(dtype=enc_attn_mask.dtype) * ner_bias  # [batch, seq_len]
            # Expand to [batch, seq_len, seq_len] so that each query sees the same key bias
            ner_bias_mask = ner_bias_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            # Repeat for each head to match enc_attn_mask shape: [batch * num_heads, seq_len, seq_len]
            num_heads = self.layers[0].self_attn.num_heads
            ner_bias_mask = ner_bias_mask.repeat_interleave(num_heads, dim=0)
            enc_attn_mask = enc_attn_mask + ner_bias_mask
        # Pass through each encoder layer
        for i, layer in enumerate(self.layers):
            # layer returns (hidden_states,) tuple
            hidden_states = layer(hidden_states, enc_attn_mask)[0]
            # Debug: print shape after each layer
            #print(f"[Encoder] layer {i} output shape: {hidden_states.shape}")
        # Return final hidden states
        return hidden_states

    def _prepare_encoder_attention_mask(self, attention_mask: torch.Tensor, batch_size: int, src_len: int):
        """
        Prepare a 3D attention mask for the encoder that matches the format expected by nn.MultiheadAttention.
        Returns a mask tensor of shape [batch * num_heads, src_len, src_len].
        """
        # Convert attention_mask (1 for tokens, 0 for padding) to a float mask where pads become 1.
        if attention_mask.dtype == torch.bool:
            inverted = ~attention_mask
            attn_mask = inverted.to(dtype=torch.float32)
        else:
            attn_mask = 1.0 - attention_mask.to(torch.float32)
        
        # Scale: padded positions get -1e9, valid positions 0.
        attn_mask = attn_mask * -1e9  # shape: [batch, src_len]
        
        # Expand so every query position gets the same key mask: [batch, 1, src_len] -> [batch, src_len, src_len]
        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, src_len, src_len)
        
        # Get number of heads (assumes at least one encoder layer exists)
        num_heads = self.layers[0].self_attn.num_heads
        
        # Repeat for each head: final shape [batch * num_heads, src_len, src_len]
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)
        return attn_mask


class CustomBartEncoderLayer(nn.Module):
    """
    Custom BART Encoder Layer with NER-enhanced self-attention (via mask bias).
    """
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = nn.MultiheadAttention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = nn.GELU() if config.activation_function in ["gelu", "gelu_new"] else nn.ReLU()
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # Self-attention
        attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)
        # Residual connection + layer norm
        hidden_states = self.self_attn_layer_norm(hidden_states + attn_output)
        # Feed-forward network
        ffn_output = self.activation_fn(self.fc1(hidden_states))
        ffn_output = nn.functional.dropout(ffn_output, p=self.activation_dropout, training=self.training)
        ffn_output = self.fc2(ffn_output)
        ffn_output = nn.functional.dropout(ffn_output, p=self.dropout, training=self.training)
        # Residual + final layer norm
        hidden_states = self.final_layer_norm(hidden_states + ffn_output)
        #print(f"hidden states :{hidden_states}")
        return (hidden_states,)
