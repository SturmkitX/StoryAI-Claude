import re

import pytest

from continuity_suite import build_messages, clean_reply, fit_history, repetition_rate, run_story, summarize
from stories import FILLER_TURNS, STORIES, STORY_LENGTHS


class FakeGenerator:
    """Treats each word as a token and answers from whatever is in its prompt."""

    def __init__(self, context_tokens=10_000):
        self.context_tokens = context_tokens

    def count_tokens(self, messages):
        return sum(len(message["content"].split()) for message in messages)

    def generate(self, messages, max_new_tokens, sample, seed=0):
        # Echoing the context means a fact is "recalled" only if it's still visible.
        return " ".join(message["content"] for message in messages[1:-1]) if not sample else "And then it got stranger."


def test_there_are_thirty_stories_with_unique_ids():
    assert len(STORIES) == 30
    assert len({story.id for story in STORIES}) == 30


def test_story_lengths_cover_every_size():
    assert {story.length for story in STORIES} == set(STORY_LENGTHS)
    for story in STORIES:
        assert len(story.turns()) == story.length
        assert story.probe_points()[-1] == story.length


@pytest.mark.parametrize("story", STORIES, ids=lambda story: story.id)
def test_answers_only_come_from_their_own_fact_turn(story):
    for index, fact in enumerate(story.facts):
        pattern = re.compile(fact.answer, re.IGNORECASE)
        assert pattern.search(fact.turn), "the fact turn must contain its answer"
        assert not pattern.search(fact.question), "the question gives away its answer"
        for other in story.facts[:index] + story.facts[index + 1 :]:
            assert not pattern.search(other.turn), f"another turn leaks {fact.answer}: {other.turn}"
        for filler in FILLER_TURNS:
            assert not pattern.search(filler), f"filler leaks {fact.answer}: {filler}"


def test_clean_reply_strips_thinking():
    assert clean_reply("<think>\nhmm\n</think>\n\nThe goat wept.") == "The goat wept."
    assert clean_reply("<think>never finished") == ""


def test_fit_history_drops_oldest_turns_first():
    history = [{"index": i, "user": "word " * 10, "reply": "word " * 10} for i in range(5)]
    kept, messages = fit_history(FakeGenerator(context_tokens=80), history, "next", reserve=10)
    assert [turn["index"] for turn in kept] == [3, 4]
    assert messages == build_messages(kept, "next")


def test_repetition_rate():
    assert repetition_rate(["one two three four five", "six seven eight nine ten"]) == 0
    assert repetition_rate(["one two three four", "one two three four"]) == 0.5
    assert repetition_rate([]) == 0


def test_run_story_tracks_recall_and_window():
    story = next(s for s in STORIES if s.length == 30)
    turns, probes, controls = run_story(FakeGenerator(context_tokens=120), story, 10, 10)

    # The transcript keeps every turn even after the window drops old ones.
    assert [turn["index"] for turn in turns] == list(range(30))
    assert not any(control["correct"] for control in controls)
    # With an echoing model, a fact is recalled exactly when it's in the window.
    assert probes and all(probe["correct"] == probe["in_window"] for probe in probes)
    assert any(probe["in_window"] for probe in probes)
    assert any(not probe["in_window"] for probe in probes)


def test_summarize_reports_each_metric():
    record = {
        "turns": [{"reply": "I'm sorry, I can't continue."}, {"reply": ""}],
        "probes": [
            {"correct": True, "in_window": True, "distance": 2},
            {"correct": False, "in_window": True, "distance": 20},
            {"correct": False, "in_window": False, "distance": 25},
        ],
        "controls": [{"correct": False}],
        "generated_tokens": 100,
        "generation_seconds": 4.0,
    }
    summary = summarize([record])
    assert summary["recall_in_window"] == {"n": 2, "recall": 0.5}
    assert summary["recall_by_distance"]["1-3"] == {"n": 1, "recall": 1.0}
    assert summary["recall_by_distance"]["16-+"] == {"n": 1, "recall": 0.0}
    assert summary["recall_out_of_window"]["n"] == 1
    assert summary["refusal_rate"] == 0.5
    assert summary["empty_reply_rate"] == 0.5
    assert summary["tokens_per_second"] == 25.0
