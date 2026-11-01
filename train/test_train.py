import pytest
from transformers import AutoTokenizer

from train_finetune import IGNORE_INDEX, assistant_prefix, encode_conversation

MESSAGES = [
    {"role": "system", "content": "Be weird."},
    {"role": "user", "content": "A goat appears."},
    {"role": "assistant", "content": "It audits you."},
    {"role": "user", "content": "Oh no."},
    {"role": "assistant", "content": "Oh yes."},
]


@pytest.fixture(params=["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen3-0.6B"])
def tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


def test_encoding_matches_the_tokens_used_at_generation_time(tokenizer):
    """Training must see exactly the prompt format that generation will use."""
    prefix = assistant_prefix(tokenizer)
    encoded = encode_conversation(MESSAGES, tokenizer, 4096, prefix)
    prompt = tokenizer.apply_chat_template(MESSAGES[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    expected = tokenizer(prompt, add_special_tokens=False).input_ids
    # Everything before the final reply must match the generation prompt exactly.
    reply_tokens = len(tokenizer("Oh yes.<|im_end|>\n", add_special_tokens=False).input_ids)
    assert encoded["input_ids"][:-reply_tokens] == expected


def test_only_replies_are_learned(tokenizer):
    encoded = encode_conversation(MESSAGES, tokenizer, 4096, assistant_prefix(tokenizer))
    learned = tokenizer.decode([label for label in encoded["labels"] if label != IGNORE_INDEX])
    assert learned == "It audits you.<|im_end|>Oh yes.<|im_end|>"


def test_truncation_keeps_inputs_and_labels_aligned(tokenizer):
    encoded = encode_conversation(MESSAGES, tokenizer, 10, "")
    assert len(encoded["input_ids"]) == len(encoded["labels"]) == 10


def test_qwen_replies_get_an_empty_thinking_block():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert "</think>" in assistant_prefix(tokenizer)
