import json

from make_review import embed_json


def test_closing_script_tag_in_data_cannot_break_out():
    out = embed_json([{"text": "evil </script><img src=x onerror=alert(1)> and S</b>trategy"}])
    assert "</script" not in out and "</" not in out
    assert json.loads(out)[0]["text"].startswith("evil </script>")  # same data after parsing
