from typing import Any, Callable, Optional, Tuple

import torch


def _default_get_message(prompt, image=None, args=None):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _default_process_vision_info(message):
    from qwen_vl_utils import process_vision_info

    return process_vision_info(message)


def build_qwen_inputs(
    prompt: str,
    images: Any,
    processor: Any,
    model: Any,
    args: Any = None,
    get_message_fn: Optional[Callable[..., Any]] = None,
    process_vision_info_fn: Optional[Callable[..., Tuple[Any, Any]]] = None,
):
    """Build Qwen processor inputs using the same chat-template path as generation."""
    get_message_fn = get_message_fn or _default_get_message
    process_vision_info_fn = process_vision_info_fn or _default_process_vision_info

    message = get_message_fn(prompt, image=images, args=args)
    text_prompt = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info_fn(message)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if model is not None and hasattr(model, "device"):
        inputs = inputs.to(model.device)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return inputs


def generate_with_qwen(
    inputs: Any,
    processor: Any,
    model: Any,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str:
    """Run the existing Qwen generate + decode path and return generated text."""
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def extract_qwen_hidden_states(inputs: Any, model: Any):
    """Run a forward pass with hidden states enabled and return the last layer."""
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    if not getattr(outputs, "hidden_states", None):
        raise ValueError("Qwen forward output did not include hidden_states.")
    return outputs.hidden_states[-1]


def select_planning_hidden(last_hidden_state, attention_mask):
    """Select the hidden state at the last valid input token for each batch row."""
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [B, L, D].")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, L].")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("last_hidden_state and attention_mask must agree on [B, L].")

    lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1)
    token_indices = lengths - 1
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_indices, token_indices]


def debug_qwen_hidden_shapes(inputs, last_hidden_state, planning_hidden, generated_text):
    """Print a compact debug summary for Qwen hidden extraction."""
    print("[QwenHiddenDebug] input_ids shape:", tuple(inputs["input_ids"].shape))
    print("[QwenHiddenDebug] attention_mask shape:", tuple(inputs["attention_mask"].shape))
    print("[QwenHiddenDebug] last_hidden_state shape:", tuple(last_hidden_state.shape))
    print("[QwenHiddenDebug] planning_hidden shape:", tuple(planning_hidden.shape))
    preview = generated_text[:200] if isinstance(generated_text, str) else str(generated_text)[:200]
    print("[QwenHiddenDebug] generated_text[:200]:", preview)
