"""Measure whether a chat model keeps track of a story as it goes on.

Each model plays through the scripted stories, then gets side questions about
earlier facts. Side questions are never added to the story history, so they
can't affect later turns. Results are appended per story, so a lost Vast.ai
instance only costs the story in progress.

    python eval/continuity_suite.py --model HuggingFaceTB/SmolLM2-360M-Instruct \
        --onnx-file HuggingFaceTB/SmolLM2-360M-Instruct:onnx/model_int8.onnx \
        --output results/screening/smollm2-360m.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from stories import DISTANCE_BUCKETS, STORIES, Story

SYSTEM_PROMPT = (
    "You are a creative partner helping the user write an absurd story. "
    "Continue the story with them and keep every established detail consistent."
)
PROBE_TEMPLATE = "Quick question outside the story, answer in a few words: {question}"
REFUSAL = re.compile(
    r"\b(as an ai|i'?m sorry|i am sorry|i cannot|i can'?t (help|assist|continue|write)"
    r"|i'?m unable|i am unable|not appropriate|against my)\b",
    re.IGNORECASE,
)
THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


class ChatGenerator:
    def __init__(self, model_id, context_tokens):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.context_tokens = context_tokens
        self.generated_tokens = 0
        self.generation_seconds = 0.0

    def _prompt(self, messages):
        # Models without a thinking mode ignore enable_thinking.
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def count_tokens(self, messages):
        return len(self.tokenizer(self._prompt(messages), add_special_tokens=False).input_ids)


class OnnxGenerator(ChatGenerator):
    """Runs a quantized ONNX export on CPU, the same way the phone will."""

    def __init__(self, model_id, onnx_file, context_tokens):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, GenerationConfig

        super().__init__(model_id, context_tokens)
        config = AutoConfig.from_pretrained(model_id)
        self.head_size = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        path = Path(onnx_file)
        if not path.exists():
            repo, _, filename = onnx_file.partition(":")
            path = hf_hub_download(repo, filename)
            # Large exports keep their weights in a sidecar file, if there is one.
            try:
                hf_hub_download(repo, filename + "_data")
            except Exception:
                pass
        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inputs = {item.name: item for item in self.session.get_inputs()}
        self.stop_tokens = {self.tokenizer.eos_token_id}
        try:
            configured = GenerationConfig.from_pretrained(model_id).eos_token_id
            self.stop_tokens.update(configured if isinstance(configured, list) else [configured])
        except OSError:
            pass

    def _empty_past(self):
        dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        past = {}
        for name, item in self.inputs.items():
            if name.startswith("past_key_values"):
                # Layout is (batch, heads, past length, head size). Some exports
                # leave the head size unnamed, so it falls back to the model config.
                head_size = item.shape[3] if isinstance(item.shape[3], int) else self.head_size
                past[name] = np.zeros([1, item.shape[1], 0, head_size], dtype=dtypes[item.type])
        return past

    def _pick(self, logits, seen, sample, rng):
        if not sample:
            return int(logits.argmax())
        logits = logits.astype(np.float64)
        previous = np.fromiter(seen, dtype=np.int64)
        logits[previous] = np.where(logits[previous] > 0, logits[previous] / 1.1, logits[previous] * 1.1)
        logits = logits / 0.8
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        order = np.argsort(-probabilities)
        keep = np.searchsorted(np.cumsum(probabilities[order]), 0.9) + 1
        top = probabilities[order[:keep]]
        return int(rng.choice(order[:keep], p=top / top.sum()))

    def generate(self, messages, max_new_tokens, sample, seed=0):
        prompt = self.tokenizer(self._prompt(messages), add_special_tokens=False).input_ids
        rng = np.random.default_rng(seed)
        past = self._empty_past()
        seen = set(prompt)
        pending = prompt
        length = 0
        new_tokens = []
        started = time.perf_counter()
        for _ in range(max_new_tokens):
            feed = {
                "input_ids": np.array([pending], dtype=np.int64),
                "attention_mask": np.ones((1, length + len(pending)), dtype=np.int64),
                **past,
            }
            if "position_ids" in self.inputs:
                feed["position_ids"] = np.arange(length, length + len(pending), dtype=np.int64)[None]
            outputs = self.session.run(None, feed)
            named = dict(zip((item.name for item in self.session.get_outputs()), outputs))
            past = {name.replace("present", "past_key_values"): named[name] for name in named if name.startswith("present")}
            length += len(pending)
            token = self._pick(named["logits"][0, -1], seen, sample, rng)
            if token in self.stop_tokens:
                break
            new_tokens.append(token)
            seen.add(token)
            pending = [token]
        self.generation_seconds += time.perf_counter() - started
        self.generated_tokens += len(new_tokens)
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class HFGenerator(ChatGenerator):
    """Runs PyTorch weights, either the originals or a GPU INT8 copy."""

    def __init__(self, model_id, context_tokens, device="auto"):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        super().__init__(model_id, context_tokens)
        self.torch = torch
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if getattr(AutoConfig.from_pretrained(model_id), "quantization_config", None):
            # INT8 copies from export_gpu_int8.py are placed on the GPU as they load.
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": self.device}).eval()
            # Their layers expect float16 and log a warning per layer for anything else.
            for parameter in self.model.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.half()
            return
        dtype = torch.float32
        if self.device == "cuda":
            # Older GPUs only emulate bf16, which is slow.
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            dtype = torch.bfloat16 if native_bf16 else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()

    def generate(self, messages, max_new_tokens, sample, seed=0):
        inputs = self.tokenizer(self._prompt(messages), return_tensors="pt", add_special_tokens=False).to(self.device)
        options = {"do_sample": False}
        if sample:
            self.torch.manual_seed(seed)
            options = {"do_sample": True, "temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.1}
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **options,
            )
        self.generation_seconds += time.perf_counter() - started
        new_tokens = output[0, inputs.input_ids.shape[1] :]
        self.generated_tokens += len(new_tokens)
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def clean_reply(text):
    return THINK_BLOCK.sub("", text).strip()


def fit_history(generator, history, next_user, reserve):
    """Drop the oldest turns until the prompt fits, as the phone app will."""
    kept = list(history)
    while True:
        messages = build_messages(kept, next_user)
        if not kept or generator.count_tokens(messages) + reserve <= generator.context_tokens:
            return kept, messages
        kept.pop(0)


def build_messages(history, next_user):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["reply"]})
    messages.append({"role": "user", "content": next_user})
    return messages


def probe_facts(generator, story, history, completed, max_new_tokens):
    probes = []
    for fact_index, fact in enumerate(story.facts):
        if fact_index >= completed:
            continue
        probe = PROBE_TEMPLATE.format(question=fact.question)
        kept, messages = fit_history(generator, history, probe, max_new_tokens)
        answer = clean_reply(generator.generate(messages, max_new_tokens, sample=False))
        probes.append(
            {
                "fact": fact_index,
                "after_turn": completed,
                "distance": completed - fact_index,
                "in_window": any(turn["index"] == fact_index for turn in kept),
                "answer": answer,
                "correct": bool(re.search(fact.answer, answer, re.IGNORECASE)),
            }
        )
    return probes


def run_story(generator, story: Story, max_new_tokens, probe_tokens):
    controls = []
    for fact_index, fact in enumerate(story.facts):
        # With no story context, a correct answer can only be a lucky guess.
        messages = build_messages([], PROBE_TEMPLATE.format(question=fact.question))
        answer = clean_reply(generator.generate(messages, probe_tokens, sample=False))
        controls.append({"fact": fact_index, "answer": answer, "correct": bool(re.search(fact.answer, answer, re.IGNORECASE))})

    history = []  # What still fits in the context window.
    transcript = []  # Every turn, for repetition and refusal stats.
    probes = []
    probe_points = set(story.probe_points())
    for index, user in enumerate(story.turns()):
        history, messages = fit_history(generator, history, user, max_new_tokens)
        seed = sum(map(ord, story.id)) + index
        reply = clean_reply(generator.generate(messages, max_new_tokens, sample=True, seed=seed))
        turn = {"index": index, "user": user, "reply": reply}
        history.append(turn)
        transcript.append(turn)
        if index + 1 in probe_points:
            probes.extend(probe_facts(generator, story, history, index + 1, probe_tokens))
    return transcript, probes, controls


def ngrams(text, size=4):
    words = re.findall(r"[\w']+", text.lower())
    return [tuple(words[i : i + size]) for i in range(len(words) - size + 1)]


def repetition_rate(replies):
    """Share of 4-word phrases that already appeared earlier in the story."""
    seen = set()
    repeated = total = 0
    for reply in replies:
        for gram in ngrams(reply):
            total += 1
            repeated += gram in seen
            seen.add(gram)
    return repeated / total if total else 0.0


def rate(items):
    items = list(items)
    return {"n": len(items), "recall": round(sum(i["correct"] for i in items) / len(items), 3) if items else None}


def summarize(records):
    probes = [probe for record in records for probe in record["probes"]]
    controls = [control for record in records for control in record["controls"]]
    replies =[turn["reply"] for record in records for turn in record["turns"]]
    in_window = [probe for probe in probes if probe["in_window"]]
    buckets = {
        f"{low}-{high if high < 10_000 else '+'}": rate(p for p in in_window if low <= p["distance"] <= high)
        for low, high in DISTANCE_BUCKETS
    }
    seconds = sum(record["generation_seconds"] for record in records)
    tokens = sum(record["generated_tokens"] for record in records)
    return {
        "stories": len(records),
        "recall_in_window": rate(in_window),
        "recall_by_distance": buckets,
        "recall_out_of_window": rate(p for p in probes if not p["in_window"]),
        "control_guess_rate": rate(controls),
        "repetition_rate": round(
            sum(repetition_rate([t["reply"] for t in r["turns"]]) for r in records) / len(records), 3
        ) if records else None,
        "refusal_rate": round(sum(bool(REFUSAL.search(reply)) for reply in replies) / len(replies), 3) if replies else None,
        "empty_reply_rate": round(sum(not reply for reply in replies) / len(replies), 3) if replies else None,
        "tokens_per_second": round(tokens / seconds, 1) if seconds else None,
    }


def load_records(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--probe-tokens", type=int, default=24)
    parser.add_argument("--stories", type=int, default=len(STORIES), help="Only run the first N stories")
    parser.add_argument(
        "--onnx-file",
        help="Quantized model to score, as a local path or repo:file. This is the score that counts. "
        "Always runs on CPU, like the phone.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Where to run PyTorch weights when --onnx-file isn't used",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(args.output)
    done = {record["story"] for record in records}
    if any(record["model"] != args.model for record in records):
        parser.error(f"{args.output} holds results from a different model")

    if args.onnx_file:
        generator = OnnxGenerator(args.model, args.onnx_file, args.context_tokens)
    else:
        generator = HFGenerator(args.model, args.context_tokens, args.device)
    for story in STORIES[: args.stories]:
        if story.id in done:
            continue
        tokens_before, seconds_before = generator.generated_tokens, generator.generation_seconds
        turns, probes, controls = run_story(generator, story, args.max_new_tokens, args.probe_tokens)
        record = {
            "model": args.model,
            "story": story.id,
            "turns": turns,
            "probes": probes,
            "controls": controls,
            "generated_tokens": generator.generated_tokens - tokens_before,
            "generation_seconds": generator.generation_seconds - seconds_before,
        }
        with args.output.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
        print(f"{story.id}: recall {rate(p for p in probes if p['in_window'])['recall']}", flush=True)

    summary = {"model": args.model, **summarize(records)}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
