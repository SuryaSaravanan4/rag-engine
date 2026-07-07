"""Tests for the unified `rag` CLI entrypoint."""
import pytest

from src import cli


@pytest.fixture(autouse=True)
def _fake_config(monkeypatch):
    """Skip real file I/O for config.yaml — every test supplies its own fake config."""
    monkeypatch.setattr(cli, "_load_config", lambda path: {"config": path})


def test_main_ingest_dispatches_with_parsed_args(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "ingest", lambda input_dir, config: captured.update(input_dir=input_dir, config=config))
    monkeypatch.setattr("sys.argv", ["rag", "ingest", "--input", "data/raw", "--config", "my.yaml"])

    cli.main()

    assert captured["input_dir"] == "data/raw"
    assert captured["config"] == {"config": "my.yaml"}


def test_main_query_prints_answer_when_not_streaming(monkeypatch, capsys):
    monkeypatch.setattr(cli, "query", lambda question, config, source=None: "the answer")
    monkeypatch.setattr("sys.argv", ["rag", "query", "What is X?"])

    cli.main()

    assert capsys.readouterr().out == "the answer\n"


def test_main_query_streams_pieces_when_stream_flag_set(monkeypatch, capsys):
    monkeypatch.setattr(cli, "stream_query", lambda question, config, source=None: iter(["the ", "answer"]))
    monkeypatch.setattr("sys.argv", ["rag", "query", "What is X?", "--stream"])

    cli.main()

    assert capsys.readouterr().out == "the answer\n"


def test_main_query_passes_source_filter_through(monkeypatch):
    captured = {}

    def fake_query(question, config, source=None):
        captured["source"] = source
        return "answer"

    monkeypatch.setattr(cli, "query", fake_query)
    monkeypatch.setattr("sys.argv", ["rag", "query", "What is X?", "--source", "subdir/file.md"])

    cli.main()

    assert captured["source"] == "subdir/file.md"


def test_main_requires_a_subcommand(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rag"])
    with pytest.raises(SystemExit):
        cli.main()
