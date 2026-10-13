"""Verify the Phocinae adapter without a running phocinae-server.

Proves the wire shape in both directions: the list-of-questions request the
server wants (choice options = the 18 move descriptions, in criteria order),
and the index + option_scores response parsed back into the arena's key-based
contract. No credentials, no network, no model — mirrors test_clef_mock.py.

Run: python test_phocinae_mock.py
"""

from __future__ import annotations

from engines import PhocinaeEngine, move_question  # noqa: E402

STATE = "a scrambled cube\nU: WWWWWWWWW ..."

ORDER = list(move_question("x")["criteria"].keys())


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def answers_body(idx: int = 3) -> dict:
    """A phocinae-server /v1/systemone response: index + option_scores."""
    rest = (1.0 - 0.62) / 17
    probs = [rest] * 18
    probs[idx] = 0.62
    return {
        "model": "Phocinae-Largha-150M-v1",
        "answers": {"move": idx},
        "usage": {"input_tokens": 412, "output_tokens": 0},
        "answer_confidence": {"move": 0.62},
        "option_scores": {"move": probs},
        "routing": {"model": "Phocinae-Largha-150M-v1", "device": "cpu",
                    "perm": "none", "backend": "phocinae-pure-torch"},
    }


def run_case(name: str, payload: dict, expect_choice: str | None, status: int = 200):
    eng = PhocinaeEngine()
    sent: dict = {}

    def fake_post(url, json=None, **kw):  # noqa: A002
        sent["url"] = url
        sent["body"] = json
        return FakeResponse(payload, status)

    eng.client.post = fake_post
    d = eng.decide(STATE, move_question("Which move?"), "move")

    ok = True
    if expect_choice is not None and d.choice != expect_choice:
        ok = False
    if expect_choice is None and not d.error:
        ok = False
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}: choice={d.choice!r} "
          f"conf={d.confidence} probs={len(d.probabilities)} err={(d.error or '')[:40]!r}")
    return ok, sent, d


def main() -> None:
    print("=== URL + request shape ===")
    eng = PhocinaeEngine()
    print("  url:", eng.url)
    assert eng.url == "http://127.0.0.1:8155/v1/systemone", eng.url
    assert eng.health_url == "http://127.0.0.1:8155/health", eng.health_url
    print("  urls ok: True")

    print("=== response parsing ===")
    all_ok = True

    # 1. standard index-based body: index 3 -> fourth criteria key
    ok, sent, d = run_case("index 3 body", answers_body(3), ORDER[3])
    all_ok &= ok
    body = sent["body"]
    assert body["model"] == "Phocinae-Largha-150M-v1", body.get("model")
    assert isinstance(body["questions"], list) and len(body["questions"]) == 1
    q = body["questions"][0]
    assert q["id"] == "move" and q["type"] == "choice"
    assert q["options"] == list(move_question("x")["criteria"].values()), \
        "options must be the criteria descriptions, in order"
    assert len(q["options"]) == 18, len(q["options"])
    print("  request: model + questions[move: choice, 18 options] ok")
    assert d.confidence == 0.62 and abs(sum(d.probabilities.values()) - 1.0) < 1e-6
    assert set(d.probabilities) == set(ORDER)

    # 2. last option index (17) maps to the last key
    ok, _, d = run_case("index 17 body", answers_body(17), ORDER[17])
    all_ok &= ok

    # 3. HTTP error surfaces cleanly
    ok, _, d = run_case("HTTP 500", {}, None, status=500)
    all_ok &= ok
    assert d.error and "HTTP" in d.error, d.error

    # 4. malformed body surfaces a parse error, never a bogus move
    ok, _, d = run_case("malformed body", {"nope": 1}, None)
    all_ok &= ok
    assert d.error and "parse" in d.error, d.error

    # 5. out-of-range index is rejected
    ok, _, d = run_case("index 99", {"answers": {"move": 99},
                                     "option_scores": {"move": [0.0] * 18}}, None)
    all_ok &= ok
    assert d.error and "parse" in d.error, d.error

    print()
    print("ALL PHOCINAE ADAPTER CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
