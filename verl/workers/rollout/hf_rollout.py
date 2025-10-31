# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib
import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, DynamicCache

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        print("Starting HFRollout")
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        # Support multiple knobs for generation micro-batching
        mb = (
            self.config.get("micro_batch_size", None)
            or self.config.get("micro_batch_size_per_gpu", None)
            or self.config.get("log_prob_micro_batch_size", None)
        )
        if mb is None:
            mb = batch_size
        num_chunks = max(batch_size // mb, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output_chunks = []
        for p in batch_prompts:
            # If guidance is provided per-sample via non_tensor_batch and this micro-batch has size 1,
            # copy the IDs into meta_info so the inner loop can access them reliably.
            try:
                if p.batch is not None and p.batch.batch_size[0] == 1:
                    gai = p.non_tensor_batch.get("guided_answer_ids", None)
                    if gai is not None and len(gai) > 0 and gai[0] is not None:
                        p.meta_info["guided_answer_ids"] = gai[0]
            except Exception:
                pass
            output_chunks.append(self._generate_minibatch(p))
        return DataProto.concat(output_chunks)

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": 1,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"] or 151643

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            guided_answer_ids = prompts.meta_info.get("guided_answer_ids", None)
            # Fallback: batched per-sample guidance passed via non_tensor_batch as object array
            if guided_answer_ids is None:
                gai = prompts.non_tensor_batch.get("guided_answer_ids", None)
                if gai is not None and len(gai) > 0 and prompts.batch.batch_size[0] == 1:
                    guided_answer_ids = gai[0]
            guided_tau = prompts.meta_info.get("guided_tau", 0.01)

            use_guidance = guided_answer_ids is not None
            if use_guidance:
                if isinstance(guided_answer_ids, list):
                    guided_answer_ids = torch.tensor([guided_answer_ids], dtype=torch.int32, device=idx.device)
                else:
                    guided_answer_ids = torch.tensor([guided_answer_ids.astype(np.int32)], dtype=torch.int32, device=idx.device)
                print("Guiding:")
                print(guided_answer_ids)

                base_prompt_ids = idx[:, :-1]
                outputs = self.module(
                    input_ids=base_prompt_ids,
                    attention_mask=attention_mask[:, :-1],
                    position_ids=position_ids[:, :-1],
                    use_cache=True
                )
                prefill_kv_cache = outputs.past_key_values
                current_completion_ids = torch.tensor(idx[:, prompt_length-1:prompt_length], dtype=torch.long, device=idx.device)
                
                for step in range(response_length):
                    answer_guided_output = self.module(
                        input_ids = torch.cat([current_completion_ids[:, -1:], guided_answer_ids], dim=1),
                        past_key_values = prefill_kv_cache,
                        use_cache=True
                    )
                    prefill_kv_cache.crop(max_length=-guided_answer_ids.shape[1])

                    logits = answer_guided_output.logits
                    probs = torch.softmax(logits, dim=-1)
                    answer_probability = 1
                    for token_idx in range(guided_answer_ids.shape[1]):
                        answer_probability *= probs[:, token_idx, guided_answer_ids[:, token_idx]]
                    
                    if answer_probability >= guided_tau:
                        seq = torch.cat([base_prompt_ids, current_completion_ids, guided_answer_ids], dim=1)
                        print("Guided answer accepted with probability:", answer_probability)
                        break
                    else:
                        new_token = torch.argmax(probs[:, 0, :], dim=-1)
                        current_completion_ids = torch.cat([current_completion_ids, new_token.unsqueeze(0)], dim=1)
                        
                    if new_token == eos_token_id or step == response_length - 1:
                        seq = torch.cat([base_prompt_ids, current_completion_ids], dim=1)
                        print(seq.shape)
                        print(seq)
                        break
            else:
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True,
                )
                seq = output.sequences

        # TODO: filter out the seq with no answers like ds-chat
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)
