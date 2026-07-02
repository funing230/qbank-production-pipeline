import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.reviewer import QwenReviewer


def sample_question(correct="D"):
    return {
        "question_text": "Which option matches the visible chart value?",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "correct_answer": correct,
        "explanation": "SECRET_EXPLANATION_SHOULD_NOT_LEAK",
        "truth_spec": "SECRET_TRUTH_SPEC_SHOULD_NOT_LEAK",
        "image_prompt": "SECRET_IMAGE_PROMPT_SHOULD_NOT_LEAK",
        "final_image_prompt": "SECRET_FINAL_IMAGE_PROMPT_SHOULD_NOT_LEAK",
        "image_contract": {"visual_facts": ["bar D has value 4"]},
        "subject": "math",
    }


def install_fake_rollout_client(monkeypatch, reviewer, outputs, prompts):
    calls = iter(outputs)

    def fake_call(prompt):
        prompts.append(prompt)
        return next(calls)

    monkeypatch.setattr(reviewer, "_call_qwen_candidate", fake_call, raising=False)


def test_rollout_prompt_whitelist_does_not_leak_answer_or_private_fields(monkeypatch):
    reviewer = QwenReviewer("https://old.example/v1", "secret", "old-model")
    prompts = []
    install_fake_rollout_client(monkeypatch, reviewer, ['{"answer":"A"}'] * 5, prompts)

    result = reviewer.review_rollouts(sample_question(correct="D"), {"kp_name": "charts"})

    assert result["decision"] == "PASS"
    assert len(prompts) == 5
    combined = "\n".join(prompts)
    assert "SECRET_EXPLANATION_SHOULD_NOT_LEAK" not in combined
    assert "SECRET_TRUTH_SPEC_SHOULD_NOT_LEAK" not in combined
    assert "SECRET_IMAGE_PROMPT_SHOULD_NOT_LEAK" not in combined
    assert "SECRET_FINAL_IMAGE_PROMPT_SHOULD_NOT_LEAK" not in combined
    assert '"correct_answer"' not in combined
    assert "correct_answer" not in combined


def test_rollout_wrong_at_least_three_passes(monkeypatch):
    reviewer = QwenReviewer("https://old.example/v1", "secret", "old-model")
    prompts = []
    install_fake_rollout_client(
        monkeypatch,
        reviewer,
        ['{"answer":"A"}', '{"answer":"B"}', '{"answer":"D"}', '{"answer":"C"}', '{"answer":"D"}'],
        prompts,
    )

    result = reviewer.review_rollouts(sample_question(correct="D"))

    assert result["decision"] == "PASS"
    assert result["source"] == "qwen_candidate_5rollout"
    assert result["wrong_count"] == 3
    assert result["correct_count"] == 2


def test_rollout_correct_at_least_three_fails(monkeypatch):
    reviewer = QwenReviewer("https://old.example/v1", "secret", "old-model")
    prompts = []
    install_fake_rollout_client(
        monkeypatch,
        reviewer,
        ['{"answer":"D"}', '{"answer":"D"}', '{"answer":"A"}', '{"answer":"D"}', '{"answer":"B"}'],
        prompts,
    )

    result = reviewer.review_rollouts(sample_question(correct="D"))

    assert result["decision"] == "FAIL"
    assert result["source"] == "qwen_candidate_5rollout"
    assert result["correct_count"] == 3
    assert result["wrong_count"] == 2


def test_rollout_valid_answers_less_than_five_is_technical_failure(monkeypatch):
    reviewer = QwenReviewer("https://old.example/v1", "secret", "old-model")
    prompts = []
    install_fake_rollout_client(
        monkeypatch,
        reviewer,
        ['{"answer":"A"}', 'not json', '{"answer":"D"}', '', '{"answer":"B"}'],
        prompts,
    )

    result = reviewer.review_rollouts(sample_question(correct="D"))

    assert result["decision"] == "FAIL"
    assert result["source"] == "qwen_technical_failure"
    assert result["technical_failure"] is True
    assert result["valid_count"] == 3
    assert result["wrong_count"] == 2
    assert result["correct_count"] == 1
