import torch
import torch.nn as nn
from transformers import BartConfig, PreTrainedModel, AutoModelForSeq2SeqLM

from .custom_encoder import CustomBartEncoder
from .custom_decoder import CustomBartDecoder
from .train_utils import compute_ner_penalty

class CustomBartModel(PreTrainedModel):
    """
    Custom BART Model with custom encoder and decoder.
    Inherits from PreTrainedModel for easy weight loading and saving.
    """
    config_class = BartConfig

    def __init__(self, config: BartConfig):
        super().__init__(config)

        #print(f"config : {config}")
        # Shared token embeddings
        self.shared = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        print(f"config:{config}")
        # Initialize shared token embeddings from normal distribution like BART
        nn.init.normal_(self.shared.weight, mean=0, std=0.02)
        # Instantiate encoder and decoder
        self.encoder = CustomBartEncoder(config, embed_tokens=self.shared)
        self.decoder = CustomBartDecoder(config, embed_tokens=self.shared)
        # Final LM head for vocabulary projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        #print(f"lm_head_vocab_size : {config.vocab_size}")
        # Tie lm_head weight to shared embedding
        self.lm_head.weight = self.shared.weight
        #print(f"lm_head_weight : {self.lm_head.weight}")


        # Initialize new parameters (section embeddings, etc.) with standard method
        self.post_init()  # from PreTrainedModel, calls init_weights

    def forward(self, input_ids=None, attention_mask=None,
                decoder_input_ids=None, decoder_attention_mask=None,
                labels=None, ner_mask=None, section_ids=None, subsection_ids=None):
        """
        Run the full model: encoder then decoder.
        If labels are provided, compute the loss (CrossEntropy + NER penalty).
        """
        # Encode the input using the custom encoder
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                      section_ids=section_ids, subsection_ids=subsection_ids,
                                      ner_mask=ner_mask)
        #print(f"encoder output :{encoder_output}")
        # Prepare encoder attention mask for decoder cross-attn if not already 4D
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                #print("atention mask 2d ")
                enc_attn_mask = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * -1e9
                #print("attenion mask converted to 4d")
            else:
                enc_attn_mask = attention_mask
        else:
            enc_attn_mask = None

        # Decode (with teacher forcing if decoder_input_ids are provided)
        decoder_outputs = self.decoder(decoder_input_ids=decoder_input_ids,
                                       decoder_attention_mask=decoder_attention_mask,
                                       encoder_hidden_states=encoder_output,
                                       encoder_attention_mask=enc_attn_mask)
        sequence_output = decoder_outputs  # shape: [batch, target_len, d_model]
        # Compute LM logits from the output of the decoder
        lm_logits = self.lm_head(sequence_output)
        #print(f"sequence_output :{sequence_output}")
        loss = None
        if labels is not None:
            # Flatten the logits and labels for calculating loss
            #print(f"labels:{labels}")
            vocab_size = lm_logits.size(-1)
            #print(f"vocab size after lm_logits :{vocab_size}")
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id,label_smoothing=0.1)
            loss_ce = loss_fct(lm_logits.view(-1, vocab_size), labels.view(-1))
            print(f"loss_ce:{loss_ce}")
            # Compute NER penalty loss if ner_mask provided
            if ner_mask is not None:
                penalty = compute_ner_penalty(input_ids, labels, self)
                
            else:
                penalty = 0.0
            loss = loss_ce + penalty
            #print(f"logits:{lm_logits}")
            return {"loss": loss, "logits": lm_logits}
            
        return {"logits": lm_logits}

    def load_pretrained_weights(self, bart_model_name="facebook/bart-base"):
        """
        Load weights from a pretrained HuggingFace Bart model into this custom model.
        Prints state dictionary summaries of both the pretrained model and the custom model
        before mapping the weights.
        """
        base_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_name)
        base_state = base_model.state_dict()
        own_state = self.state_dict()

        # Print summary of pretrained state dictionary
        print("Pretrained model state dictionary summary:")
        for key, tensor in base_state.items():
            print(f"{key}: {list(tensor.size())}")

        print("\nCustom model initial state dictionary summary:")
        for key, tensor in own_state.items():
            print(f"{key}: {list(tensor.size())}")

        # Copy encoder weights using concatenation of Q, K, V projections
        for i in range(self.config.encoder_layers):
            q_proj = base_state[f"model.encoder.layers.{i}.self_attn.q_proj.weight"]
            k_proj = base_state[f"model.encoder.layers.{i}.self_attn.k_proj.weight"]
            v_proj = base_state[f"model.encoder.layers.{i}.self_attn.v_proj.weight"]
            in_proj_weight = torch.cat([q_proj, k_proj, v_proj], dim=0)
            own_state[f"encoder.layers.{i}.self_attn.in_proj_weight"] = in_proj_weight

            q_bias = base_state[f"model.encoder.layers.{i}.self_attn.q_proj.bias"]
            k_bias = base_state[f"model.encoder.layers.{i}.self_attn.k_proj.bias"]
            v_bias = base_state[f"model.encoder.layers.{i}.self_attn.v_proj.bias"]
            in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            own_state[f"encoder.layers.{i}.self_attn.in_proj_bias"] = in_proj_bias

            own_state[f"encoder.layers.{i}.self_attn.out_proj.weight"] = base_state[f"model.encoder.layers.{i}.self_attn.out_proj.weight"]
            own_state[f"encoder.layers.{i}.self_attn.out_proj.bias"] = base_state[f"model.encoder.layers.{i}.self_attn.out_proj.bias"]
            own_state[f"encoder.layers.{i}.self_attn_layer_norm.weight"] = base_state[f"model.encoder.layers.{i}.self_attn_layer_norm.weight"]
            own_state[f"encoder.layers.{i}.self_attn_layer_norm.bias"] = base_state[f"model.encoder.layers.{i}.self_attn_layer_norm.bias"]
            own_state[f"encoder.layers.{i}.fc1.weight"] = base_state[f"model.encoder.layers.{i}.fc1.weight"]
            own_state[f"encoder.layers.{i}.fc1.bias"] = base_state[f"model.encoder.layers.{i}.fc1.bias"]
            own_state[f"encoder.layers.{i}.fc2.weight"] = base_state[f"model.encoder.layers.{i}.fc2.weight"]
            own_state[f"encoder.layers.{i}.fc2.bias"] = base_state[f"model.encoder.layers.{i}.fc2.bias"]
            own_state[f"encoder.layers.{i}.final_layer_norm.weight"] = base_state[f"model.encoder.layers.{i}.final_layer_norm.weight"]
            own_state[f"encoder.layers.{i}.final_layer_norm.bias"] = base_state[f"model.encoder.layers.{i}.final_layer_norm.bias"]

        # Copy decoder weights using concatenation for self-attention and direct mapping for cross-attention
        for j in range(self.config.decoder_layers):
            q_proj = base_state[f"model.decoder.layers.{j}.self_attn.q_proj.weight"]
            k_proj = base_state[f"model.decoder.layers.{j}.self_attn.k_proj.weight"]
            v_proj = base_state[f"model.decoder.layers.{j}.self_attn.v_proj.weight"]
            in_proj_weight = torch.cat([q_proj, k_proj, v_proj], dim=0)
            own_state[f"decoder.layers.{j}.self_attn.in_proj_weight"] = in_proj_weight

            q_bias = base_state[f"model.decoder.layers.{j}.self_attn.q_proj.bias"]
            k_bias = base_state[f"model.decoder.layers.{j}.self_attn.k_proj.bias"]
            v_bias = base_state[f"model.decoder.layers.{j}.self_attn.v_proj.bias"]
            in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            own_state[f"decoder.layers.{j}.self_attn.in_proj_bias"] = in_proj_bias

            own_state[f"decoder.layers.{j}.self_attn.out_proj.weight"] = base_state[f"model.decoder.layers.{j}.self_attn.out_proj.weight"]
            own_state[f"decoder.layers.{j}.self_attn.out_proj.bias"] = base_state[f"model.decoder.layers.{j}.self_attn.out_proj.bias"]
            own_state[f"decoder.layers.{j}.self_attn_layer_norm.weight"] = base_state[f"model.decoder.layers.{j}.self_attn_layer_norm.weight"]
            own_state[f"decoder.layers.{j}.self_attn_layer_norm.bias"] = base_state[f"model.decoder.layers.{j}.self_attn_layer_norm.bias"]

            own_state[f"decoder.layers.{j}.cross_attn_k_proj.weight"] = base_state[f"model.decoder.layers.{j}.encoder_attn.k_proj.weight"]
            own_state[f"decoder.layers.{j}.cross_attn_k_proj.bias"] = base_state[f"model.decoder.layers.{j}.encoder_attn.k_proj.bias"]
            own_state[f"decoder.layers.{j}.cross_attn_v_proj.weight"] = base_state[f"model.decoder.layers.{j}.encoder_attn.v_proj.weight"]
            own_state[f"decoder.layers.{j}.cross_attn_v_proj.bias"] = base_state[f"model.decoder.layers.{j}.encoder_attn.v_proj.bias"]

            for idx in range(4):
                own_state[f"decoder.layers.{j}.cross_attn_q_projs.{idx}.weight"] = base_state[f"model.decoder.layers.{j}.encoder_attn.q_proj.weight"]
                own_state[f"decoder.layers.{j}.cross_attn_q_projs.{idx}.bias"] = base_state[f"model.decoder.layers.{j}.encoder_attn.q_proj.bias"]

            own_state[f"decoder.layers.{j}.cross_attn_layer_norm.weight"] = base_state[f"model.decoder.layers.{j}.encoder_attn_layer_norm.weight"]
            own_state[f"decoder.layers.{j}.cross_attn_layer_norm.bias"] = base_state[f"model.decoder.layers.{j}.encoder_attn_layer_norm.bias"]
            own_state[f"decoder.layers.{j}.fc1.weight"] = base_state[f"model.decoder.layers.{j}.fc1.weight"]
            own_state[f"decoder.layers.{j}.fc1.bias"] = base_state[f"model.decoder.layers.{j}.fc1.bias"]
            own_state[f"decoder.layers.{j}.fc2.weight"] = base_state[f"model.decoder.layers.{j}.fc2.weight"]
            own_state[f"decoder.layers.{j}.fc2.bias"] = base_state[f"model.decoder.layers.{j}.fc2.bias"]
            own_state[f"decoder.layers.{j}.final_layer_norm.weight"] = base_state[f"model.decoder.layers.{j}.final_layer_norm.weight"]
            own_state[f"decoder.layers.{j}.final_layer_norm.bias"] = base_state[f"model.decoder.layers.{j}.final_layer_norm.bias"]

        own_state["shared.weight"] = base_state["model.shared.weight"]
        own_state["encoder.embed_positions.weight"] = base_state["model.encoder.embed_positions.weight"]
        own_state["decoder.embed_positions.weight"] = base_state["model.decoder.embed_positions.weight"]
        own_state["encoder.layernorm_embedding.weight"] = base_state["model.encoder.layernorm_embedding.weight"]
        own_state["encoder.layernorm_embedding.bias"] = base_state["model.encoder.layernorm_embedding.bias"]
        own_state["decoder.layernorm_embedding.weight"] = base_state["model.decoder.layernorm_embedding.weight"]
        own_state["decoder.layernorm_embedding.bias"] = base_state["model.decoder.layernorm_embedding.bias"]
        own_state["lm_head.weight"] = base_state["model.shared.weight"]

        self.load_state_dict(own_state)
        print("Loaded pretrained BART weights into CustomBartModel.")
def ngram_in_sequence(ngram, seq):
    """
    Check if the given ngram (tuple of token IDs) exists as a contiguous subsequence in seq (a list of token IDs).
    
    Args:
        ngram (tuple): An n-gram (e.g. (id1, id2, id3)).
        seq (list): A list of token IDs.
    
    Returns:
        bool: True if ngram is found in seq; otherwise False.
    """
    n = len(ngram)
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i+n]) == ngram:
            return True
    return False


class CustomBartForConditionalGeneration(CustomBartModel):
    def generate_text(self, input_ids, attention_mask, section_ids, subsection_ids, ner_mask,
                      max_length=512, beam_size=5, no_repeat_ngram_size=3, repetition_penalty=1.2,
                      length_penalty=1.0, early_stopping=True, min_length=125):
        """
        Generate text output using beam search with options to mitigate repetition and force a minimum length.
        
        Args:
            input_ids, attention_mask, section_ids, subsection_ids, ner_mask: inputs to the encoder.
            max_length (int): Maximum decoding length.
            beam_size (int): Number of beams to maintain.
            no_repeat_ngram_size (int): Blocks repeated n-grams.
            repetition_penalty (float): Penalty factor for already generated tokens.
            length_penalty (float): Normalizes beam scores by length.
            early_stopping (bool): Stop once enough beams have finished.
            min_length (int): Minimum number of tokens to generate before EOS is allowed.
        
        Returns:
            str: Generated text decoded from token IDs.
        """
        self.eval()
        with torch.no_grad():
            # Encode the input once.
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                section_ids=section_ids,
                subsection_ids=subsection_ids,
                ner_mask=ner_mask
            )
            bos_token_id = self.config.bos_token_id 
            #print(f"bos token id in the generate function  :{self.config.bos_token_id} ")
            eos_token_id = self.config.eos_token_id
            pad_token_id = self.config.pad_token_id

            # Initialize beam search: list of tuples (sequence, cumulative_log_prob)
            beams = [([bos_token_id], 0.0)]
            #print(f"initial beam :{beams}")
            finished_beams = []
            #print(f"max length :{max_length}")
            for step in range(max_length):
                #print(f"step :{step}")
                new_beams = []
                for seq, seq_score in beams:
                    # If last token is EOS and sequence length >= min_length, mark beam as finished.
                    if seq[-1] == eos_token_id:
                        #print("in the continue vala loop seq")
                        norm_score = seq_score / (len(seq) ** length_penalty) if length_penalty != 1.0 else seq_score
                        finished_beams.append((seq, norm_score))
                        #print("in continue")
                        continue

                    # Prepare decoder input (current sequence).
                    current_seq = torch.tensor([seq], dtype=torch.long, device=input_ids.device)
                    #print(f"current seq :{current_seq}")
                    dec_mask = (current_seq != pad_token_id).long()
                    enc_attn_mask = None
                    if attention_mask is not None:
                        enc_attn_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9

                    # Run decoder.
                    #print("b4 decoder ")
                    decoder_outputs = self.decoder(
                        decoder_input_ids=current_seq,
                        decoder_attention_mask=dec_mask,
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=enc_attn_mask
                    )
                    #print(f"decoder output:{decoder_outputs}")
                    hidden_states = decoder_outputs  # shape: [1, seq_len, d_model]
                    logits = self.lm_head(hidden_states)  # shape: [1, seq_len, vocab_size]
                    next_token_logits = logits[:, -1, :]  # logits for last token [1, vocab_size]

                    # Force minimum length: if current seq length < min_length, block EOS.
                    if len(seq) < min_length:
                        next_token_logits[0, eos_token_id] = -float('inf')

                    # Apply repetition penalty.
                    if repetition_penalty != 1.0:
                        for token_id in set(seq):
                            if token_id in (bos_token_id, eos_token_id, pad_token_id):
                                continue
                            token_val = next_token_logits[0, token_id]
                            if token_val > 0:
                                next_token_logits[0, token_id] = token_val / repetition_penalty
                            else:
                                next_token_logits[0, token_id] = token_val * repetition_penalty

                    # Compute log probabilities.
                    log_probs = torch.log_softmax(next_token_logits, dim=-1).squeeze(0)
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

                    # Expand beams.
                    for cand_id, cand_log_prob in zip(topk_ids.tolist(), topk_log_probs.tolist()):
                        # Check no-repeat n-gram constraint.
                        if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                            n = no_repeat_ngram_size
                            ngram = tuple(seq[-(n-1):] + [cand_id]) if n > 1 else (cand_id,)
                            if ngram_in_sequence(ngram, seq):
                                continue
                        new_seq = seq + [cand_id]
                        new_score = seq_score + cand_log_prob
                        new_beams.append((new_seq, new_score))
                # Fallback: if no new beams were generated, force expansion by choosing best non-EOS candidate.
                if not new_beams:
                    for seq, seq_score in beams:
                        if seq[-1] == eos_token_id:
                            print("niche vale continue mai!")
                            #continue
                        current_seq = torch.tensor([seq], dtype=torch.long, device=input_ids.device)
                        dec_mask = (current_seq != pad_token_id).long()
                        enc_attn_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9 if attention_mask is not None else None
                        decoder_outputs = self.decoder(
                            decoder_input_ids=current_seq,
                            decoder_attention_mask=dec_mask,
                            encoder_hidden_states=encoder_output,
                            encoder_attention_mask=enc_attn_mask
                        )
                        hidden_states = decoder_outputs
                        logits = self.lm_head(hidden_states)
                        next_token_logits = logits[:, -1, :]
                        if len(seq) < min_length:
                            next_token_logits[0, eos_token_id] = -float('inf')
                        if repetition_penalty != 1.0:
                            for token_id in set(seq):
                                if token_id in (bos_token_id, eos_token_id, pad_token_id):
                                    continue
                                token_val = next_token_logits[0, token_id]
                                if token_val > 0:
                                    next_token_logits[0, token_id] = token_val / repetition_penalty
                                else:
                                    next_token_logits[0, token_id] = token_val * repetition_penalty
                        log_probs = torch.log_softmax(next_token_logits, dim=-1).squeeze(0)
                        sorted_ids = torch.argsort(log_probs, descending=True).tolist()
                        forced_cand = None
                        for cand in sorted_ids:
                            if cand != eos_token_id:
                                forced_cand = cand
                                break
                        if forced_cand is not None:
                            new_seq = seq + [forced_cand]
                            new_score = seq_score + log_probs[forced_cand].item()
                            new_beams.append((new_seq, new_score))
                    if not new_beams:
                        break
                # Keep only top beam_size new beams.
                new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                beams = new_beams

                if early_stopping and len(finished_beams) >= beam_size:
                    break

            if not finished_beams:
                finished_beams = [(seq, seq_score / (len(seq) ** length_penalty)) for seq, seq_score in beams]

            best_seq, _ = sorted(finished_beams, key=lambda x: x[1], reverse=True)[0]
            # Remove BOS token.
            output_ids = best_seq[1:] if best_seq[0] == bos_token_id else best_seq
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return generated_text
