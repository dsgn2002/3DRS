#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config
from .qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM, log_3d_losses
from llava.model.three_d_supervision import contrastive_supervision_enabled, supervision_setting

import torch.distributed as dist
try:
    from deepspeed.comm import get_rank
except Exception:  # noqa: BLE001 - analysis/inference without the training stack.
    # Not just ImportError: a deepspeed present but unable to find CUDA_HOME
    # raises MissingCUDAException from its op builder at import time.
    def get_rank():
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def gather_loss(loss):
    # 将标量 loss 转为张量（确保设备一致）
    loss_tensor = torch.tensor([loss], device=torch.cuda.current_device())

    # 收集所有卡的 loss 张量
    world_size = dist.get_world_size()
    gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_losses, loss_tensor)

    # 转换为数值列表
    gathered_losses = [loss.item() for loss in gathered_losses]
    return gathered_losses


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if hasattr(config, "ground_head_type") and config.ground_head_type is not None:
            self.ground_head_type = config.ground_head_type
            if config.ground_head_type == "mlp":
                # self.ground_head = nn.Sequential(
                #     nn.Linear(config.hidden_size, config.ground_head_hidden_size),
                #     nn.ReLU(),
                #     nn.LayerNorm(config.ground_head_hidden_size),x
                #     nn.Linear(config.ground_head_hidden_size, 6)
                # )
                self.ground_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size)
                )
            elif config.ground_head_type == "score":
                self.ground_head_temperature = config.ground_head_temperature
                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_score = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1),
                )
            elif config.ground_head_type == "infonce":
                # self.ground_head_temperature = nn.Parameter(torch.tensor(config.ground_head_temperature))
                try:
                    self.ground_head_temperature = config.ground_head_temperature
                except:
                    self.ground_head_temperature = 0.07
                self.ground_head_zero_target = torch.nn.Parameter(torch.randn(config.hidden_size))

                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
            else:
                raise NotImplementedError
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        video_dict=None,
        use_object_proposals: bool = False,
        box_labels = None,
        img_pos_list = None,
        img_length_list = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, object_features, object_boxes, img_pos_list, img_length_list) = \
                self.prepare_inputs_labels_for_multimodal(
                    input_ids, 
                    position_ids, 
                    attention_mask, 
                    past_key_values, 
                    labels, 
                    images, 
                    modalities, 
                    image_sizes, 
                    video_dict,
                    use_object_proposals=use_object_proposals,
                )
            #print(img_pos_list, img_length_list)
        if use_object_proposals:
            return self.predict_box(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                object_features=object_features,
                object_boxes=object_boxes,
                box_labels=box_labels,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                video_dict=video_dict,
            )
            

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                video_dict=video_dict
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                video_dict=video_dict
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, _, _, img_pos_list, img_length_list) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, video_dict=kwargs.get("video_dict", None))
            kwargs['img_pos_list'] = img_pos_list
            kwargs['img_length_list'] = img_length_list
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        img_pos_list = kwargs.pop("img_pos_list", None)
        img_length_list = kwargs.pop("img_length_list", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if img_pos_list is not None:
            inputs["img_pos_list"] = img_pos_list
        if img_length_list is not None:
            inputs["img_length_list"] = img_length_list

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    def feature_3d_alignment(self, video_dict, hidden_states, img_pos_list=None, img_length_list=None, margin=0.0):
        """Backwards-compatible scalar view of the 3D supervision.

        Kept so external scripts that called the original single-term method
        still work; new code should use ``three_d_supervision_losses`` and log
        the terms separately.
        """
        losses = self.three_d_supervision_losses(
            video_dict, hidden_states, img_pos_list=img_pos_list, img_length_list=img_length_list
        )
        return sum(losses.values())

    def extract_object_feature(self, video_dict, hidden_states, feature_3d=None, img_pos_list=None, img_length_list=None, box_labels=None):
        object_boxes = video_dict["objects"][0]
        object_boxes_center = object_boxes[:, :3]

        obj_num = len(object_boxes)

        object_patch = []
        # ignore the batch dimension here
        world_coords = video_dict["world_coords"][0]
        C = hidden_states.shape[-1]
        image_features = hidden_states[:,img_pos_list[0]:img_pos_list[0]+img_length_list[0],:].view(32,14,15,C)[:,:,:-1,:].contiguous().view(32, 14*14,C)
        # image_features = hidden_states[:,img_pos_list[0]:img_pos_list[0]+img_length_list[0],:].view(32*196+1,C)[:-1,:].contiguous().view(32, 14*14,C)
        feature_3d = feature_3d.view(32, 14*14, 2048) if feature_3d is not None else feature_3d

        for l in range(obj_num):
            box = object_boxes[l]
            min_xyz = box[:3] - box[3:] / 2
            max_xyz = box[:3] + box[3:] / 2
            
            world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 14, 27, 14, 27, 3).transpose(2, 3).flatten(3, 4)  # [32, 14, 14, 27*27, 3]
            cur_object_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)     # [32, 14, 14, 27*27]
            cur_object_patch = cur_object_patch.sum(dim=3) >= int(27 * 27 * 0.125)
            object_patch.append(cur_object_patch)
        
        object_features = []
        ground_head_features_3d = []
        for l in range(obj_num):
            cur_object_features = image_features[object_patch[l].view(-1, 196)]
            
            if len(cur_object_features) == 0:
                cur_object_features = torch.zeros(image_features.shape[-1]).to(image_features[0].device)
            else:
                cur_object_features = cur_object_features.mean(dim=0)

            if isinstance(box_labels, list) and len(box_labels[0]) > 0 and l in box_labels[0] and feature_3d is not None:
                ground_head_features_3d.append(feature_3d[object_patch[l].view(-1, 196)].mean(dim=0))

            object_features.append(cur_object_features)
        object_features = torch.stack(object_features)
        ground_head_features_3d = torch.stack(ground_head_features_3d) if len(ground_head_features_3d) > 0 else ground_head_features_3d

        return object_features, ground_head_features_3d
    
    def predict_box(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        cache_position=None,
        video_dict=None,
        object_features=None,
        object_boxes=None,
        box_labels=None,
        img_pos_list=None,
        img_length_list=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # Grounding samples (ScanRefer/Multi3DRefer, ~36% of the mix) take this
        # path, so an intermediate-layer contrast term needs the layers here too.
        need_layers = (
            box_labels is not None
            and contrastive_supervision_enabled(self.config)
            and supervision_setting(self.config, "three_d_contrast_layer") != -1
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states or need_layers,
            return_dict=return_dict or need_layers,
            img_pos_list=img_pos_list,
            img_length_list=img_length_list
        )

        hidden_states = outputs[0]
        all_hidden_states = outputs.hidden_states if need_layers else None
        ground_locations = (labels >= self.config.ground_token_ids[0]) & (labels <= self.config.ground_token_ids[-1])
        ground_hidden = hidden_states[ground_locations].squeeze(1)
        object_features_llm, _ = self.extract_object_feature(video_dict, hidden_states, feature_3d=None, img_pos_list=img_pos_list, img_length_list=img_length_list, box_labels=box_labels) 
        object_features = (object_features + object_features_llm) / 2

        if self.ground_head_type == 'mlp':
            ground_hidden = self.ground_head(ground_hidden).squeeze(0) 
            scores = (ground_hidden * object_features).sum(dim=-1)
        elif self.ground_head_type == 'score':
            obj_feat = self.ground_head_obj(object_features.to(ground_hidden.dtype)) # B, C
            query_feat = self.ground_head_query(ground_hidden) # 1, C
            # sim = (F.normalize(obj_feat) * F.normalize(query_feat)).sum(dim=-1)
            mul_feat = obj_feat * query_feat
            scores = self.ground_head_score(mul_feat) # B, 1
            scores = scores.squeeze(1)

        elif self.ground_head_type == "infonce":
            object_features = torch.cat([object_features, self.ground_head_zero_target.unsqueeze(0)], dim=0)
            obj_feat = self.ground_head_obj(object_features.to(ground_hidden.dtype))
            query_feat = self.ground_head_query(ground_hidden)
            obj_feat = F.normalize(obj_feat)
            query_feat = F.normalize(query_feat)
            scores = (obj_feat * query_feat).sum(dim=-1)

        loss = None
        if box_labels is not None:
            if self.ground_head_type == "infonce":
                if len(box_labels[0]) == 0: # zero-target
                    box_labels[0].append(-1)
                logits = torch.exp(scores / self.ground_head_temperature)
                loss = - torch.log( logits[box_labels[0]].sum() / logits.sum())
                # negative_logits_sum = logits.sum() - logits[box_labels[0]].sum()
                # for idx in box_labels[0]:
                #     loss += - torch.log(logits[idx] / (negative_logits_sum + logits[idx]))
                # loss /= len(box_labels[0])
            else:
                bce_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                target = torch.zeros_like(scores)
                target[box_labels[0]] = 1
                weight = torch.ones_like(scores)
                if len(box_labels[0]) != 0:
                    weight[box_labels[0]] *= (scores.shape[0] - len(box_labels[0])) / len(box_labels[0])
                
                bce_loss = (bce_loss_fct(scores, target.detach()) * weight).mean()
                loss = bce_loss  
                # nce_loss = 0
                # logits = torch.exp(sim / self.ground_head_temperature)
                # negative_logits_sum = logits.sum() - logits[box_labels[0]].sum()
                # if len(box_labels[0]) != 0:
                #     for idx in box_labels[0]:
                #         nce_loss += - torch.log(logits[idx] / (negative_logits_sum + logits[idx]))
                #     nce_loss /= len(box_labels[0])
                # loss = bce_loss + nce_loss
            if loss is not None:
                aux_losses = self.three_d_supervision_losses(
                    video_dict, hidden_states, img_pos_list=img_pos_list,
                    img_length_list=img_length_list, all_hidden_states=all_hidden_states,
                )
                for value in aux_losses.values():
                    loss = loss + value.to(loss.dtype)
                log_3d_losses(aux_losses, prefix="Grounding")
        
        return loss, scores

        # loss = None
        # if box_labels is not None:
        #     ## BCE
        #     loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        #     target = torch.zeros_like(scores)
        #     target[box_labels[0]] = 1
        #     weight = torch.ones_like(scores)
        #     weight[box_labels[0]] *= scores.shape[0] - 1
        #     loss = (loss_fct(scores, target.detach()) * weight).mean()
        #     ## CE 
        #     # loss_fct = nn.CrossEntropyLoss()
        #     # loss = loss_fct(scores, box_labels[0]) / self.config.ground_loss_scale


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)

