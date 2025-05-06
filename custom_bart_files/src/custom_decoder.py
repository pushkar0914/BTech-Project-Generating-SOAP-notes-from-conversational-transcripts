import torch
import torch.nn as nn
import math
from transformers import BartConfig

class CustomBartDecoder(nn.Module):
    """
    Custom BART Decoder with multi-query cross-attention for SOAP sections.
    - Uses 4 separate cross-attention query projections (Subjective, Objective, Assessment, Plan).
    - Shares key/value projections for encoder outputs.
    - Implements inter-section attention: Assessment attends to Subjective+Objective outputs, Plan attends to Assessment output.
    - Fuses the 4 attention heads via a learned weight vector.
    """
    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.config = config
        #print(f"[decoder] config :{config}")
        self.embed_tokens = embed_tokens  # shared token embeddings
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id

        # Positional embeddings (same length as encoder)
        self.embed_positions = nn.Embedding(config.max_position_embeddings + 2, embed_dim)
        nn.init.normal_(self.embed_positions.weight, mean=0, std=0.02)
        self.embed_positions.weight.data[self.padding_idx] = 0

        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.dropout = config.dropout

        # Decoder layers
        self.layers = nn.ModuleList([CustomBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.gradient_checkpointing = False  # not used here

    def forward(self, decoder_input_ids=None, decoder_attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=False):
        """
        Forward pass for the decoder.
        Args:
            decoder_input_ids (torch.LongTensor): [batch, target_len] decoder input token IDs.
            decoder_attention_mask (torch.Tensor): [batch, target_len] mask for decoder inputs (1 for tokens, 0 for padding).
            encoder_hidden_states (torch.FloatTensor): [batch, enc_len, d_model] encoder outputs to attend to.
            encoder_attention_mask (torch.Tensor): [batch, 1, 1, enc_len] encoder attention mask (already prepared).
            past_key_values: Not used (for caching in generation).
            use_cache (bool): Not used (caching not implemented in this custom model).
        Returns:
            hidden_states (torch.FloatTensor): [batch, target_len, d_model] decoder outputs.
        """
        if decoder_input_ids is None:
            raise ValueError("decoder_input_ids must be provided for decoding.")
        batch_size, target_len = decoder_input_ids.shape
        # Debug:
        #print(f"[Decoder] input shape: { (batch_size, target_len) }")

        # Embed tokens and positions
        # Note: BART uses start token as decoder_input_ids[:,0]; ensure decoder_input_ids are correctly shifted outside.
        inputs_embeds = self.embed_tokens(decoder_input_ids)
        #print(f"decoder input ids : {decoder_input_ids}")
        embed_scale = torch.sqrt(torch.tensor(self.config.d_model, dtype=torch.float)) if getattr(self.config, "scale_embedding", False) else 1.0
        inputs_embeds = inputs_embeds * embed_scale
        #print("in decoder ")
        positions = torch.arange(target_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, target_len)
        positions = positions + 2  # offset
        pos_embeds = self.embed_positions(positions)
        hidden_states = inputs_embeds + pos_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Prepare decoder attention mask (combine with causal mask)
        if decoder_attention_mask is None:
            print(f"decoder atention msk is none")
            decoder_attention_mask = torch.ones((batch_size, target_len), device=hidden_states.device)
        # Create causal mask: lower triangular matrix of shape [target_len, target_len]
        causal_mask = torch.tril(torch.ones((target_len, target_len), device=hidden_states.device))
        # Process decoder_attention_mask: convert to additive mask of shape [batch, target_len]
        if decoder_attention_mask.dtype == torch.bool:
            dec_mask = (~decoder_attention_mask).to(dtype=torch.float32)
        else:
            dec_mask = 1.0 - decoder_attention_mask.to(torch.float32)
        dec_mask = dec_mask * -1e9  # shape [batch, target_len]
        # Expand dec_mask to shape [batch, target_len, target_len]
        dec_mask = dec_mask.unsqueeze(1).expand(batch_size, target_len, target_len)
        # Combine with causal mask: positions beyond current token get -inf
        combined_mask = dec_mask + (1 - causal_mask) * -1e9  # shape: [batch, target_len, target_len]

        # FIX: Repeat the mask for each attention head
        num_heads = self.layers[0].self_attn.num_heads
        #print(f"decoder num heads : {num_heads}")
        combined_mask = combined_mask.repeat_interleave(num_heads, dim=0)  # now shape [batch*num_heads, target_len, target_len]

        # Iterate over decoder layers
        for j, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, combined_mask, encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
            # Debug shape after each layer
           # print(f"[Decoder] layer {j} output shape: {hidden_states.shape}")
        return hidden_states

class CustomBartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        # Self-attention (decoder) - uses causal masking
        self.self_attn = nn.MultiheadAttention(self.embed_dim, config.decoder_attention_heads,
                                               dropout=config.attention_dropout, batch_first=True)
        self.dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # Cross-attention projections (multi-query architecture)
        
        # Shared key and value projection for encoder hidden states
        self.cross_attn_k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        #print(f"self.cross_attn_k_proj : {self.cross_attn_k_proj}")
        self.cross_attn_v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        #print(f"self.cross_attn_v_proj : {self.cross_attn_v_proj}")
        # Separate query projections for each SOAP section
        self.num_sections = 4  # [S, O, A, P]
        self.cross_attn_q_projs = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _ in range(self.num_sections)])
        #print(f"self.cross_attn_q_projs :{self.cross_attn_q_projs}")
        # Weight vector to fuse the 4 attention outputs
        self.cross_attn_fuse_weight = nn.Parameter(torch.full((self.num_sections,), 0.25))
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        #new
        self.norm_A_enhanced = nn.LayerNorm(self.embed_dim)
        self.norm_P_enhanced = nn.LayerNorm(self.embed_dim)
        # Feed-forward network
        self.activation_fn = nn.GELU() if config.activation_function in ["gelu", "gelu_new"] else nn.ReLU()
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, combined_attn_mask: torch.Tensor,
                encoder_hidden_states: torch.Tensor, encoder_attn_mask: torch.Tensor):
        # Self-Attention (decoder)
        self_attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attn_mask=combined_attn_mask)
        self_attn_output = nn.functional.dropout(self_attn_output, p=self.dropout, training=self.training)
        #print(f"self_attn_output  :{self_attn_output }")
        hidden_states = self.self_attn_layer_norm(hidden_states + self_attn_output)

        # Cross-Attention (encoder-decoder multi-query)
        # Project encoder hidden states to key and value (shared for all queries)
        # encoder_hidden_states: [batch, enc_len, d_model]
        key_states = self.cross_attn_k_proj(encoder_hidden_states)  # [batch, enc_len, d_model]
        #print(f"key_states :{key_states}")
        value_states = self.cross_attn_v_proj(encoder_hidden_states)  # [batch, enc_len, d_model]
        #print(f"value_states:{value_states}")
        # Reshape for multi-head attention computation
        batch_size, enc_len, _ = key_states.size()
        num_heads = None
        head_dim = None
        if hasattr(self, "self_attn") and isinstance(self.self_attn, nn.MultiheadAttention):
            num_heads = self.self_attn.num_heads
            #print(f"num_heads_dec : {num_heads}")
            head_dim = self.embed_dim // num_heads
        else:
            num_heads = 1
            head_dim = self.embed_dim
        key_states = key_states.view(batch_size, enc_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, enc_len, num_heads, head_dim).transpose(1, 2)

        context_outputs = []  # will store context for S,O,A,P
        attn_mask = encoder_attn_mask  # shape [batch, 1, 1, enc_len]
        if attn_mask is not None and attn_mask.size(-1) == enc_len:
            if attn_mask.size(1) == 1 and attn_mask.size(2) == 1:
                attn_mask = attn_mask.expand(batch_size, num_heads, hidden_states.size(1), enc_len)
        else:
            #print("attn_mask_dec none")
            attn_mask = None

        for i, q_proj in enumerate(self.cross_attn_q_projs):
            query_states = q_proj(hidden_states)  # [batch, tgt_len, d_model]
            query_states = query_states.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            attn_scores = torch.matmul(query_states, key_states.transpose(2, 3))
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask
            attn_probs = nn.functional.softmax(attn_scores, dim=-1)
            attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_probs, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
            context_outputs.append(attn_output)
        enhanced_context_outputs = context_outputs[:]  # copy
        #print(enhanced_context_outputs)

        if len(context_outputs) == 4:
            S_ctx, O_ctx, A_ctx, P_ctx = context_outputs
           # print(f"Shape of S_ctx: {S_ctx.shape}")
            #print(f"Shape of O_ctx: {O_ctx.shape}")
            #print(f"Shape of A_ctx: {A_ctx.shape}")
            #print(f"Shape of P_ctx: {P_ctx.shape}")
            combined_SO = torch.cat([S_ctx, O_ctx], dim=1)
            #print(f"Shape of combined_SO: {combined_SO.shape}")
            Q_A = A_ctx
            #print(f"Shape of Q_A: {Q_A.shape}")

            K_SO = combined_SO
            #print(f"Shape of K_SO: {K_SO.shape}")
            V_SO = combined_SO
           # print(f"Shape of V_SO: {V_SO.shape}")

            attn_scores_A = torch.matmul(Q_A, K_SO.transpose(1, 2)) / math.sqrt(self.embed_dim)
            #print(f"Shape of attn_scores_A: {attn_scores_A.shape}")

            attn_weights_A = nn.functional.softmax(attn_scores_A, dim=-1)
            #print(f"Shape of attn_weights_A: {attn_weights_A.shape}")

            A_enhanced = torch.matmul(attn_weights_A, V_SO)
            #print(f"Shape of A_enhanced: {A_enhanced.shape}")

            enhanced_context_outputs[2] = self.norm_A_enhanced(A_enhanced)
            Q_P = P_ctx
            #print(f"Shape of Q_P: {Q_P.shape}")

            K_A = A_enhanced
            #print(f"Shape of K_A: {K_A.shape}")

            V_A = A_enhanced
            #print(f"Shape of V_A: {V_A.shape}")

            attn_scores_P = torch.matmul(Q_P, K_A.transpose(1, 2)) / math.sqrt(self.embed_dim)
            #print(f"Shape of attn_scores_P: {attn_scores_P.shape}")

            attn_weights_P = nn.functional.softmax(attn_scores_P, dim=-1)
           # print(f"Shape of attn_weights_P: {attn_weights_P.shape}")

            P_enhanced = torch.matmul(attn_weights_P, V_A)
           # print(f"Shape of P_enhanced: {P_enhanced.shape}")

            enhanced_context_outputs[3] =  self.norm_P_enhanced(P_enhanced)
           # print(f"Shape of enhanced_context_outputs[3] (P_enhanced): {enhanced_context_outputs[3].shape}")


        fused = 0
        weights = self.cross_attn_fuse_weight  # shape [4]
        for idx, context in enumerate(enhanced_context_outputs):
            fused = fused + weights[idx] * context
        fused_output = nn.functional.dropout(fused, p=self.dropout, training=self.training)
        #aprint(f"fused output : {fused_output}")
        hidden_states = self.cross_attn_layer_norm(hidden_states + fused_output)

        ffn_output = self.activation_fn(self.fc1(hidden_states))
        ffn_output = nn.functional.dropout(ffn_output, p=self.activation_dropout, training=self.training)
        ffn_output = self.fc2(ffn_output)
        ffn_output = nn.functional.dropout(ffn_output, p=self.dropout, training=self.training)
        hidden_states = self.final_layer_norm(hidden_states + ffn_output)
        return (hidden_states,)
 