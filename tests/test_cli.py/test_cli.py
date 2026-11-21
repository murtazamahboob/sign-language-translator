import json

from signlang.cli import main
from signlang.poses import generate


def test_alphabet_command(capsys):
    assert main(["alphabet"]) == 0
    out = capsys.readouterr().out
    assert "A" in out
    assert "HELLO" in out


def test_classify_command_json(tmp_path, capsys):
    path = tmp_path / "b.json"
    path.write_text(json.dumps(generate("B")))
    assert main(["classify", "--json", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["letter"] == "B"


def test_spell_command(tmp_path, capsys):
    path = tmp_path / "frames.jsonl"
    t = 0.0
    lines = []
    for letter in "DOG":
        for _ in range(3):
            lines.append(json.dumps({"landmarks": generate(letter), "t": t}))
            t += 0.1
        lines.append(json.dumps({"landmarks": None, "t": t}))
        t += 1.2
    path.write_text("\n".join(lines))
    assert main(["spell", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "DOG"


def test_demo_and_benchmark_commands(capsys):
    assert main(["demo"]) == 0
    assert "26/26" in capsys.readouterr().out
    assert main(["benchmark", "--seeds", "2", "--jitter", "0.01"]) == 0
    assert "Alphabet benchmark" in capsys.readouterr().out
