from __future__ import annotations

import re

from agent_reflect.promptkit import ERROR_MAX, DataTags, error_text, load_template, model_text, nonce, squash

SECRET = "ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


def test_load_template_drops_only_a_leading_attribution_comment():
    reflect = load_template("reflect.md").template
    assert not reflect.startswith("<!--") and reflect.startswith("Ты разбираешь")
    assert load_template("consolidate_plan.md").template.startswith("Ты проводишь")


def test_data_tags_defuse_only_their_own_tags():
    tags = DataTags(("a_b", "c"))
    assert tags.defuse("<a_b> </C id=1> <a_bc> <d>") == "&lt;a_b> &lt;/C id=1> <a_bc> <d>"
    assert tags.one_per_line(["  x\n y ", "", "<c>"]) == "x y\n&lt;c>"
    assert tags.one_per_line([]) == "—"


def test_model_text_squash_error_text_and_nonce():
    assert model_text(None) == "" and model_text(5) == ""
    assert model_text(" a\ud800 ") == "a?"
    assert model_text(f"key {SECRET}") == "key [REDACTED]"
    assert squash(" a \n\t b ") == "a b"
    message = error_text(RuntimeError(SECRET + " " + "x" * 400))
    assert len(message) <= ERROR_MAX == 300 and "ghp_" not in message
    assert re.fullmatch(r"[0-9a-f]{8}", nonce()) and nonce() != nonce()
