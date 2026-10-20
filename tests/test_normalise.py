"""Alias table, fuzzy fallback and method-family grouping."""
from litreview.normalise import Normaliser, clean_surface


def test_alias_hits_are_exact_and_case_insensitive():
    n = Normaliser()
    assert n.normalise("cnn/dm", "datasets") == ("CNN/DailyMail", "alias")
    assert n.normalise("RougeL", "metrics")[0] == "ROUGE-L"
    assert n.normalise("Multi-XScience", "datasets")[0] == "Multi-XScience"


def test_fuzzy_fallback_for_unlisted_surface_form():
    n = Normaliser()
    canon, how = n.normalise("BERT-Score", "metrics")
    assert canon == "BERTScore" and how.startswith(("alias", "fuzzy"))


def test_unmatched_names_are_kept_and_recorded():
    n = Normaliser()
    canon, how = n.normalise("Zorblax-9000 benchmark", "datasets")
    assert canon == "Zorblax-9000 benchmark" and how == "unmatched"
    assert "Zorblax-9000 benchmark" in n.unmatched["datasets"]
    assert n.report()["unmatched_counts"]["datasets"] == 1


def test_clean_surface_strips_trailing_year_and_punctuation():
    assert clean_surface("  XSum (2018).  ") == "XSum"


def test_method_family_grouping():
    n = Normaliser()
    assert n.method_family("bart-large") == "Pretrained seq2seq"
    assert n.method_family("scibert") == "Encoder-only transformer"
    assert n.method_family("crewai") == "Agentic / multi-agent"
    assert n.method_family("a totally unknown technique") == "Other"


def test_normalise_record_adds_norm_fields_without_losing_originals():
    n = Normaliser()
    out = n.normalise_record({
        "paper_id": "p", "title": "t", "methods": ["bart-large", "GPT-4"],
        "datasets": ["cnn/dm"], "task": "abstractive summarisation",
        "metrics": [{"name": "rouge1", "value": 44.0, "setting": "cnn/dm"}]})
    assert out["methods"] == ["bart-large", "GPT-4"]
    assert out["methods_norm"] == ["BART", "GPT-family LLM"]
    assert out["datasets_norm"] == ["CNN/DailyMail"]
    assert out["task_norm"] == "Abstractive summarization"
    assert out["metrics_norm"][0]["name_norm"] == "ROUGE-1"
    assert out["metrics_norm"][0]["setting_dataset"] == "CNN/DailyMail"
    assert set(out["method_families"]) == {"Pretrained seq2seq", "Large language model (prompted)"}


def test_deduplicates_after_normalisation():
    n = Normaliser()
    assert n.normalise_list(["cnn/dm", "CNN/Daily Mail", "cnndm"], "datasets") == ["CNN/DailyMail"]
