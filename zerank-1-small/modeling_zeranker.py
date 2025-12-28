from sentence_transformers import CrossEncoder as _CE

import math
from typing import cast, Any
import types

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

MODEL_PATH = "zeroentropy/zerank-1-small"
PER_DEVICE_BATCH_SIZE_TOKENS = 15_000
global_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def format_pointwise_datapoints(
    tokenizer: PreTrainedTokenizerFast,
    query_documents: list[tuple[str, str]],
) -> BatchEncoding:
    input_texts: list[str] = []
    for query, document in query_documents:
        system_prompt = f"""
{query}
""".strip()
        user_message = f"""
{document}
""".strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert isinstance(input_text, str)
        input_texts.append(input_text)

    batch_inputs = tokenizer(
        input_texts,
        padding=True,
        return_tensors="pt",
    )
    return batch_inputs


def load_model(
    device: torch.device | None = None,
) -> tuple[
    PreTrainedTokenizerFast,
    LlamaForCausalLM
    | Gemma3ForConditionalGeneration
    | Gemma3ForCausalLM
    | Qwen3ForCausalLM,
]:
    if device is None:
        device = global_device

    config = AutoConfig.from_pretrained(MODEL_PATH)
    assert isinstance(config, PretrainedConfig)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        quantization_config=None,
        device_map={"": device},
    )
    if config.model_type == "llama":
        model.config.attn_implementation = "flash_attention_2"
    assert isinstance(
        model,
        LlamaForCausalLM
        | Gemma3ForConditionalGeneration
        | Gemma3ForCausalLM
        | Qwen3ForCausalLM,
    )

    tokenizer = cast(
        AutoTokenizer,
        AutoTokenizer.from_pretrained(
            MODEL_PATH,
            padding_side="right",
        ),
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def predict(
    self,
    query_documents: list[tuple[str, str]] | None = None,
    *,
    sentences: Any = None,
    batch_size: Any = None,
    show_progress_bar: Any = None,
    activation_fn: Any = None,
    apply_softmax: Any = None,
    convert_to_numpy: Any = None,
    convert_to_tensor: Any = None,
) -> list[float]:
    if query_documents is None:
        if sentences is None:
            raise ValueError("query_documents or sentences must be provided")
        query_documents = [[sentence[0], sentence[1]] for sentence in sentences]

    if not hasattr(self, "inner_model"):
        self.inner_tokenizer, self.inner_model = load_model(global_device)
        self.inner_model.gradient_checkpointing_enable()
        self.inner_model.eval()
        self.inner_yes_token_id = self.inner_tokenizer.encode(
            "Yes", add_special_tokens=False
        )[0]

    model = self.inner_model
    tokenizer = self.inner_tokenizer

    query_documents = [
        (query[:2_000], document[:10_000]) for query, document in query_documents
    ]
    # Sort
    permutation = list(range(len(query_documents)))
    permutation.sort(
        key=lambda i: -len(query_documents[i][0]) - len(query_documents[i][1])
    )
    query_documents = [query_documents[i] for i in permutation]

    # Extract document batches from this line of datapoints
    max_length = 0
    batches: list[list[tuple[str, str]]] = []
    for query, document in query_documents:
        if (
            len(batches) == 0
            or (len(batches[-1]) + 1) * max(max_length, len(query) + len(document))
            > PER_DEVICE_BATCH_SIZE_TOKENS
        ):
            batches.append([])
            max_length = 0

        batches[-1].append((query, document))
        max_length = max(max_length, 20 + len(query) + len(document))

    # Inference all of the document batches
    all_logits: list[float] = []
    for batch in batches:
        batch_inputs = format_pointwise_datapoints(
            tokenizer,
            batch,
        )

        batch_inputs = batch_inputs.to(global_device)

        try:
            outputs = model(**batch_inputs, use_cache=False)
        except torch.OutOfMemoryError:
            print(f"GPU OOM! {torch.cuda.memory_reserved()}")
            torch.cuda.empty_cache()
            print(f"GPU After OOM Cache Clear: {torch.cuda.memory_reserved()}")
            outputs = model(**batch_inputs, use_cache=False)

        # Extract the logits
        logits = cast(torch.Tensor, outputs.logits)
        attention_mask = cast(torch.Tensor, batch_inputs.attention_mask)
        last_positions = attention_mask.sum(dim=1) - 1

        batch_size = logits.shape[0]
        batch_indices = torch.arange(batch_size, device=global_device)
        last_logits = logits[batch_indices, last_positions]

        yes_logits = last_logits[:, self.inner_yes_token_id]
        all_logits.extend([float(logit) / 5.0 for logit in yes_logits])

    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    scores = [sigmoid(logit) for logit in all_logits]

    # Unsort by indices
    scores = [score for _, score in sorted(zip(permutation, scores, strict=True))]

    return scores


def to_device(self: _CE, new_device: torch.device) -> None:
    global global_device
    global_device = new_device


_CE.predict = predict

from transformers import Qwen3Config

ZEConfig = Qwen3Config

_CE.to = to_device
