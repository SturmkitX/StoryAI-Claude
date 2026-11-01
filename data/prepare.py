"""Turn the raw roleplay datasets into multi-turn training conversations.

The tone of the data is left alone. Filtering is about continuity (replies that
reuse what's established) and removing boilerplate refusals. The only content
exclusion is sexual content involving minors.

    python data/prepare.py --raw data/raw --output data/processed
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from continuity_suite import SYSTEM_PROMPT  # noqa: E402

MAX_WINDOW_CHARS = 3600  # About 1,000 tokens.
MAX_MESSAGE_CHARS = 1500  # Long posts ramble; tangents are funnier when short.
MAX_WINDOWS_PER_CONVERSATION = 8
MAX_SCENARIO_CHARS = 600
MIN_MESSAGES = 4
MIN_CONTINUITY = 0.5
RECALL_EXAMPLE_RATE = 0.5
VALIDATION_PERCENT = 5

BOILERPLATE_REFUSAL = re.compile(
    r"as an ai\b|language model|cannot (fulfill|assist)|can'?t assist|content polic|"
    r"against (my|the) guidelines|not able to (generate|create) (that|such)",
    re.IGNORECASE,
)
MINOR_TERMS = re.compile(
    r"\b(child|children|kid|kids|minor|minors|underage|loli|lolita|shota|preteen|schoolgirl|schoolboy|"
    r"teenager|teen|middle school|high school|elementary school|little girl|little boy|toddler|"
    r"(?:[1-9]|1[0-7])[- ]?(?:years?[- ]?old|yo|y/o))\b",
    re.IGNORECASE,
)
SEXUAL_TERMS = re.compile(
    r"\b(sex|sexual|sexually|nude|naked|cock|penis|vagina|pussy|orgasm|cum|moan|moans|moaning|"
    r"breast|breasts|nipple|nipples|erotic|fuck|fucks|fucked|fucking|arousal|aroused)\b",
    re.IGNORECASE,
)
PLACEHOLDERS = {"{{user}}": "You", "<USER>": "You"}
NAME = re.compile(r"(?<![.!?\"*\n] )(?<!^)\b[A-Z][a-z]{2,}\b")
NOT_NAMES = {"The", "This", "That", "There", "They", "She", "Her", "His", "Him", "You", "Your", "And", "But", "What", "When", "Where", "Who", "Why", "How", "Then", "With", "For", "Not", "Yes", "Well"}
CAPITALIZED = re.compile(r"\b[A-Z][a-z]{2,}\b")
SIMPLE_FACT = re.compile(r"\b([A-Z][a-z]{2,}) (is|was) (an?|the) ([a-z][a-z' -]{2,40}?)[.,!]")
CALLED_FACT = re.compile(r"\b([Aa]n?) ([a-z][a-z' -]{2,30}?) (?:named|called) ([A-Z][a-z]{2,})\b")
RECALL_QUESTIONS = (
    "Wait, remind me: who is {name}?",
    "Hold on, I forgot. Who's {name} again?",
    "Before we go on, who is {name}?",
    "Sorry, who was {name}?",
)


def clean_text(text, bot_name=None):
    for placeholder, value in PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    if bot_name:
        text = text.replace("{{char}}", bot_name).replace("<BOT>", bot_name)
    return text.replace("<START>", "").strip()


def normalize(raw_messages):
    """Merge same-speaker runs so roles alternate, starting from the user."""
    messages = []
    for role, content in raw_messages:
        if not content:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    if messages and messages[0]["role"] == "assistant":
        # PIPPA conversations open with the character's greeting.
        messages.insert(0, {"role": "user", "content": "Let's start the story."})
    return messages


def cut_at_refusal(messages):
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and BOILERPLATE_REFUSAL.search(message["content"]):
            return messages[:index]
    return messages


def excluded_for_minors(scenario, messages):
    """Over-removes on purpose: any co-occurrence in a conversation drops all of it."""
    text = "\n".join([scenario or ""] + [message["content"] for message in messages])
    minor = MINOR_TERMS.search(text)
    return bool(minor and SEXUAL_TERMS.search(text))


def windows(messages):
    """Overlapping slices that start on a user turn and end on a reply."""
    found = []
    start = 0
    while start < len(messages) and len(found) < MAX_WINDOWS_PER_CONVERSATION:
        if messages[start]["role"] != "user":
            start += 1
            continue
        end = start
        chars = 0
        while end < len(messages):
            size = len(messages[end]["content"])
            if size > MAX_MESSAGE_CHARS or chars + size > MAX_WINDOW_CHARS:
                break
            chars += size
            end += 1
        end -= (end - start) % 2
        if end - start >= MIN_MESSAGES:
            found.append(messages[start:end])
        if end >= len(messages) - 1:
            break
        start += max(2, (end - start) // 2 // 2 * 2)
    return found


def names_in(text):
    return {name for name in NAME.findall(text) if name not in NOT_NAMES}


def continuity_score(window):
    """Share of later replies that reuse a name from earlier in the conversation.

    A word only becomes a name when seen mid-sentence, which rules out most
    capitalised sentence openers. After that it also counts at a sentence start.
    """
    seen = names_in(window[0]["content"])
    reused = replies = 0
    for index, message in enumerate(window[1:], start=1):
        if message["role"] == "assistant" and index > 1:
            replies += 1
            reused += bool(seen & set(CAPITALIZED.findall(message["content"])))
        seen |= names_in(message["content"])
    return reused / replies if replies else 0.0


def recall_example(window, rng):
    """Extend a window with a side question answered from an earlier statement."""
    facts = []
    for message in window[:-2]:
        for name, _, article, description in SIMPLE_FACT.findall(message["content"]):
            if name not in NOT_NAMES:
                facts.append((name, article, description.strip()))
        for article, description, name in CALLED_FACT.findall(message["content"]):
            if name not in NOT_NAMES:
                facts.append((name, article.lower(), description.strip()))
    if not facts:
        return None
    name, article, description = rng.choice(facts)
    return window + [
        {"role": "user", "content": rng.choice(RECALL_QUESTIONS).format(name=name)},
        {"role": "assistant", "content": f"{name} is {article} {description}."},
    ]


def conversation_key(messages):
    """The same conversation can appear in several datasets, so key it by content."""
    text = " ".join(message["content"][:300] for message in messages[1:3]).lower()
    return hashlib.sha256(re.sub(r"\W+", " ", text).encode()).hexdigest()


def load_pippa(path):
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            bot = row.get("bot_name") or "Character"
            scenario = clean_text(row.get("bot_description") or "", bot)
            raw = [("user" if m["is_human"] else "assistant", clean_text(m["message"], bot)) for m in row["conversation"]]
            yield "pippa", scenario, raw


def load_sharegpt(path, source):
    roles = {"human": "user", "gpt": "assistant"}
    for row in json.loads(path.read_text(encoding="utf-8")):
        scenario = ""
        raw = []
        for message in row["conversations"]:
            if message["from"] == "system":
                scenario = clean_text(message["value"])
            elif message["from"] in roles:
                raw.append((roles[message["from"]], clean_text(message["value"])))
        yield source, scenario, raw


def load_all(raw_dir):
    yield from load_pippa(raw_dir / "pippa" / "pippa_deduped.jsonl")
    yield from load_sharegpt(raw_dir / "bluemoon" / "data" / "full" / "bluemoon.train.json", "bluemoon")
    yield from load_sharegpt(raw_dir / "creative" / "Creative-Writing-MultiturnV1.1.json", "creative")


def system_message(scenario):
    content = SYSTEM_PROMPT
    if scenario:
        content += "\n\nScenario: " + scenario[:MAX_SCENARIO_CHARS]
    return {"role": "system", "content": content}


def process(conversations, seed=0):
    """Returns (examples by split, stats). Stats hold counts only, never text."""
    rng = random.Random(seed)
    stats = Counter()
    seen_conversations = set()
    splits = {"train": [], "validation": []}
    for source, scenario, raw in conversations:
        stats[f"{source}_conversations"] += 1
        messages = cut_at_refusal(normalize(raw))
        if len(messages) < MIN_MESSAGES:
            stats["dropped_too_short"] += 1
            continue
        if excluded_for_minors(scenario, messages):
            stats["dropped_minor_exclusion"] += 1
            continue
        key = conversation_key(messages)
        if key in seen_conversations:
            stats["dropped_duplicate"] += 1
            continue
        seen_conversations.add(key)
        split = "validation" if int(key, 16) % 100 < VALIDATION_PERCENT else "train"
        for window in windows(messages):
            if continuity_score(window) < MIN_CONTINUITY:
                stats["windows_low_continuity"] += 1
                continue
            chosen = window
            if rng.random() < RECALL_EXAMPLE_RATE:
                extended = recall_example(window, rng)
                if extended:
                    chosen = extended
                    stats["recall_examples"] += 1
            splits[split].append({"source": source, "conversation": key[:16], "messages": [system_message(scenario)] + chosen})
            stats[f"{source}_windows"] += 1
            stats[f"{split}_windows"] += 1
    return splits, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    splits, stats = process(load_all(args.raw))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as output:
            for example in examples:
                output.write(json.dumps(example, ensure_ascii=False) + "\n")
    report = dict(sorted(stats.items()))
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
