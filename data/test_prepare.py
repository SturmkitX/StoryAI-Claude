import random

from prepare import (
    MAX_WINDOW_CHARS,
    clean_text,
    continuity_score,
    conversation_key,
    cut_at_refusal,
    excluded_for_minors,
    normalize,
    process,
    recall_example,
    windows,
)


def chat(*contents):
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": content} for i, content in enumerate(contents)]


def test_clean_text_fills_placeholders():
    assert clean_text("<START>{{char}} waves at {{user}}", "Mirela") == "Mirela waves at You"


def test_normalize_merges_runs_and_starts_with_user():
    messages = normalize([("assistant", "Hi."), ("user", "One."), ("user", "Two."), ("assistant", ""), ("assistant", "Ok.")])
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[2]["content"] == "One.\n\nTwo."


def test_refusal_boilerplate_cuts_conversation_but_in_character_apology_stays():
    kept = chat("Hello", "I'm sorry, my lord, I cannot find the goat!", "Why?", "As an AI, I can't do that.", "Oh", "More")
    assert len(cut_at_refusal(kept)) == 3


def test_adult_content_stays_and_only_the_combination_is_excluded():
    assert not excluded_for_minors("", chat("They had sex.", "It was a disaster."))
    assert not excluded_for_minors("", chat("The kid stole a goat.", "The goat was furious."))
    assert excluded_for_minors("A 15 year old character.", chat("They had sex.", "Then what?"))


def test_windows_start_on_user_end_on_reply_and_respect_limits():
    messages = chat(*[f"Message {i} " + "x" * 300 for i in range(100)])
    found = windows(messages)
    assert len(found) > 1
    for window in found:
        assert window[0]["role"] == "user" and window[-1]["role"] == "assistant"
        assert sum(len(m["content"]) for m in window) <= MAX_WINDOW_CHARS


def test_windows_terminate_on_oversized_messages():
    messages = chat("hi", "x" * 5000, "hi", "x" * 5000, "a", "b", "c", "d")
    assert windows(messages) == [messages[4:8]]


def test_continuity_score_rewards_reusing_names():
    good = chat("Then Mirela arrived.", "She waved.", "Next?", "So then Mirela left.")
    bad = chat("Then Mirela arrived.", "She waved.", "Next?", "Nothing happened.")
    assert continuity_score(good) == 1.0
    assert continuity_score(bad) == 0.0


def test_recall_example_answers_from_an_earlier_statement():
    window = chat("We met Mirela. Mirela is a crab.", "Lovely.", "Next?", "Onwards.")
    extended = recall_example(window, random.Random(0))
    assert "Mirela" in extended[-2]["content"]
    assert extended[-1]["content"] == "Mirela is a crab."
    assert recall_example(chat("a", "b", "c", "d"), random.Random(0)) is None
    called = recall_example(chat("His friend is a crab called Mirela.", "b", "c", "d"), random.Random(0))
    assert called[-1]["content"] == "Mirela is a crab."


def test_duplicates_across_datasets_are_dropped_and_stay_in_one_split():
    raw = [("user", "Then Mirela arrived."), ("assistant", "So Mirela waved."), ("user", "Next?"), ("assistant", "Mirela left.")]
    splits, stats = process([("pippa", "", raw), ("creative", "", raw)])
    assert stats["dropped_duplicate"] == 1
    assert sum(len(examples) for examples in splits.values()) == 1
    example = next(iter(splits["train"] + splits["validation"]))
    assert example["messages"][0]["role"] == "system"
    assert conversation_key(example["messages"][1:])


def test_process_reports_only_counts():
    _, stats = process([("pippa", "", [("user", "x"), ("assistant", "y")])])
    assert stats["dropped_too_short"] == 1
    assert all(isinstance(value, int) for value in stats.values())
