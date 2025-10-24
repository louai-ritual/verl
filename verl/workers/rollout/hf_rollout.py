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

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

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
                # Trainer already repeats the batch by rollout.n. Avoid double-multiplying memory.
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
        pad_token_id = prompts.meta_info["pad_token_id"]

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
            use_guidance = guided_answer_ids is not None and guided_tau is not None
            if use_guidance and idx.size(0) == 1 and kwargs.get("num_return_sequences", 1) == 1:
                print("Guiding:")
                print(guided_answer_ids)
                print(guided_tau)

                input_ids = idx
                am = attention_mask
                pos = position_ids
                guided_ptr = 0
                guidance_failed = 0
                started_guidance = False
                start_snapshot = None
                max_steps = response_length
                eos_list = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
                for step in range(max_steps):
                    out = self.module(input_ids=input_ids, attention_mask=am, position_ids=pos, use_cache=False)
                    logits = out.logits[:, -1, :]
                    logits_float = logits.float()
                    probs_full = torch.softmax(logits_float, dim=-1)

                    attempt_guidance = True

                    if attempt_guidance:
                        if not started_guidance:
                            started_guidance = True
                            start_snapshot = (input_ids, am, pos)
                        target_token = guided_answer_ids[guided_ptr]
                        if probs_full[0, target_token] >= guided_tau:
                            next_token = torch.tensor([[target_token]], device=input_ids.device, dtype=input_ids.dtype)
                            guided_ptr += 1
                        else:
                            guidance_failed += 1
                            guided_ptr = 0
                            input_ids, am, pos = start_snapshot
                            out = self.module(input_ids=input_ids, attention_mask=am, position_ids=pos, use_cache=False)
                            logits = out.logits[:, -1, :]
                            logits_float = logits.float()
                            if not do_sample:
                                next_token = torch.argmax(logits_float, dim=-1, keepdim=True)
                            else:
                                l = logits_float
                                if temperature is not None and temperature != 1.0:
                                    l = l / max(temperature, 1e-8)
                                if top_k is not None and top_k > 0:
                                    topk_vals, topk_idx = torch.topk(l, k=min(top_k, l.size(-1)), dim=-1)
                                    l_masked = torch.full_like(l, float("-inf"))
                                    l_masked.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                                    l = l_masked
                                if top_p is not None and 0.0 < top_p < 1.0:
                                    sorted_logits, sorted_idx = torch.sort(l, descending=True, dim=-1)
                                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                                    cumprobs = torch.cumsum(sorted_probs, dim=-1)
                                    keep = cumprobs <= top_p
                                    keep[..., :1] = True
                                    sorted_logits = torch.where(
                                        keep, sorted_logits, torch.full_like(sorted_logits, float("-inf"))
                                    )
                                    l = torch.full_like(l, float("-inf"))
                                    l.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
                                probs = torch.softmax(l, dim=-1)
                                next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        if not do_sample:
                            next_token = torch.argmax(logits_float, dim=-1, keepdim=True)
                        else:
                            l = logits_float
                            if temperature is not None and temperature != 1.0:
                                l = l / max(temperature, 1e-8)
                            if top_k is not None and top_k > 0:
                                topk_vals, topk_idx = torch.topk(l, k=min(top_k, l.size(-1)), dim=-1)
                                l_masked = torch.full_like(l, float("-inf"))
                                l_masked.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                                l = l_masked
                            if top_p is not None and 0.0 < top_p < 1.0:
                                sorted_logits, sorted_idx = torch.sort(l, descending=True, dim=-1)
                                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                                keep = cumprobs <= top_p
                                keep[..., :1] = True
                                sorted_logits = torch.where(
                                    keep, sorted_logits, torch.full_like(sorted_logits, float("-inf"))
                                )
                                l = torch.full_like(l, float("-inf"))
                                l.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
                            probs = torch.softmax(l, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)

                    input_ids = torch.cat((input_ids, next_token), dim=1)
                    am = torch.cat((am, torch.ones((am.size(0), 1), dtype=am.dtype, device=am.device)), dim=1)
                    pos = torch.cat((pos, pos[:, -1:] + 1), dim=1)

                    if next_token[0, 0].item() in eos_list:
                        break
                    if started_guidance and guided_ptr == len(guided_answer_ids):
                        print('Answer Guidance Succeeded!!')
                        break

                seq = input_ids
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
