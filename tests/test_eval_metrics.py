"""Citation coverage/correctness and the bootstrap, with a stub NLI scorer."""
from litreview.eval import metrics as M


class StubNLI:
    """Entails only when the hypothesis shares a distinctive word with the premise."""

    def score(self, pairs):
        out = []
        for prem, hyp in pairs:
            out.append((0.9 if "marmoset" in prem.lower() and "marmoset" in hyp.lower() else 0.05,
                        0.0))
        return out


def test_coverage_counts_ids_and_title_mentions():
    gold = [{"cite_key": "see2017", "title": None},
            {"cite_key": "liu2019", "title": "Fine tuning BERT encoders for summarization"},
            {"cite_key": "absent", "title": "Something nobody mentioned at all here"}]
    pred_text = ("Pointer networks were introduced [see2017]. Later work fine tuning BERT "
                 "encoders for summarization pushed this further.")
    sents = [{"text": pred_text, "cited_paper_ids": ["see2017"]}]
    cov = M.citation_coverage(pred_text, sents, gold)
    assert cov["n_discussed"] == 2 and cov["n_gold_cited"] == 3
    assert cov["coverage"] == 0.6667 or abs(cov["coverage"] - 2 / 3) < 1e-3
    assert cov["detail"][0]["by_id"] and cov["detail"][1]["by_title"]


def test_correctness_penalises_a_sentence_attached_to_the_wrong_paper():
    sents = [{"text": "The marmoset result holds.", "cited_paper_ids": ["p1", "p2"]}]
    ev = {"p1": ["we observe the marmoset effect clearly"], "p2": ["an unrelated finding"]}
    cc = M.citation_correctness(sents, ev, StubNLI(), threshold=0.3)
    assert cc["n_checkable"] == 2 and cc["correctness"] == 0.5


def test_correctness_reports_unresolvable_ids_separately():
    sents = [{"text": "A claim.", "cited_paper_ids": ["ghost"]}]
    cc = M.citation_correctness(sents, {"p1": ["x"]}, StubNLI())
    assert cc["n_unresolvable_paper_ids"] == 1 and cc["n_checkable"] == 0
    assert cc["correctness"] == 0.0 and cc["correctness_strict"] == 0.0


def test_bootstrap_ci_brackets_the_mean_and_handles_tiny_n():
    ci = M.bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_boot=500, seed=1)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"] and ci["n"] == 5
    assert M.bootstrap_ci([0.4])["ci_low"] is None
    assert M.bootstrap_ci([])["mean"] is None


def test_aggregate_skips_missing_metrics():
    rows = [{"rougeL": 0.3}, {"rougeL": 0.5}, {}]
    agg = M.aggregate(rows, ["rougeL", "bertscore"])
    assert agg["rougeL"]["n"] == 2 and agg["bertscore"]["n"] == 0
