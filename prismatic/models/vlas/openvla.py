"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

    # TODO: Remove the add_space_in_output parameter when no longer necessary. Currently present because Will messed up the training data 
    # in one of his runs :(
    @torch.inference_mode()
    def predict_action(
        self, image: Union[Img, List[Img]], instruction: str, unnorm_key: Optional[str] = None, 
        enforce_cot=False, add_space_in_output=True, add_period_at_end=False, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer

        instruction = instruction.lower().strip()
        if add_period_at_end:
            if not instruction.endswith("."):
                instruction += "."

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"{instruction.lower()}")
        if enforce_cot:
            prompt_builder.add_turn(role="gpt", message=enforce_cot)
            prompt_text = prompt_builder.get_prompt()

            # When giving a prefix for the output, it auto-adds the EOS token, so must manually remove.
            if prompt_text.endswith("</s>"):
                prompt_text = prompt_text.split("</s>")[0]
        else:
            prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if add_space_in_output and not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            # do nothing here. I think...
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")
        
        # print(input_ids)

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            if "max_new_tokens" not in kwargs.keys():
                kwargs["max_new_tokens"] = self.get_action_dim(unnorm_key)
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                # max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on

        
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, generated_ids.cpu().numpy()


    @torch.inference_mode()
    def predict_rationale(
        self, image, prompt_text = None, instruction = None, unnorm_key = None,
        add_space_in_output=True, enforce_cot=False, return_action=True, **kwargs
    ):
        image_transform, tokenizer = self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer

        assert not (prompt_text is None and instruction is None), "Either `prompt_text` or `instruction` must be provided."

        if prompt_text is None:
            assert instruction is not None, "If `prompt_text` is not provided, `instruction` must be provided."
            instruction = instruction.lower().strip()
            if not instruction.endswith("."):
                instruction += "."

            # Build VLA Prompt
            prompt_builder = self.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=f"{instruction.lower()}")

            if enforce_cot:
                prompt_builder.add_turn(role="gpt", message=enforce_cot)
                prompt_text = prompt_builder.get_prompt()

                # When giving a prefix for the output, it auto-adds the EOS token, so must manually remove.
                if prompt_text.endswith("</s>"):
                    prompt_text = prompt_text.split("</s>")[0]
            else:
                prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if add_space_in_output and not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            # do nothing here. I think...
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            if "max_new_tokens" not in kwargs.keys():
                kwargs["max_new_tokens"] = self.get_action_dim(unnorm_key)
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                # max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on

        if not return_action:
            return None, generated_ids

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, generated_ids


    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
