"""Contradiction detection and gap candidate generation, on hand-built records."""
from litreview.gaps import build_matrix, candidate_gaps
from litreview.synthesis import build_graph, consensus_map, contradictions_table, find_contradictions


def rec(pid, task, ds, metric, value, methods, year=2024, families=None):
    return {"paper_id": pid, "title": f"paper {pid}", "year": year,
            "task_norm": task, "datasets_norm": [ds],
            "methods_norm": methods, "method_families": families or ["Pretrained seq2seq"],
            "claims": [], "metrics_norm": [
                {"name_norm": metric, "value": value, "setting_dataset": ds,
                 "higher_is_better": True}]}


def test_agreeing_values_are_not_flagged():
    recs = [rec("a", "Summ", "XSum", "ROUGE-L", 40.0, ["BART"]),
            rec("b", "Summ", "XSum", "ROUGE-L", 41.0, ["BART"])]
    cm = consensus_map(recs)
    assert cm["n_groups"] == 1 and cm["groups"][0]["agrees"]
    assert find_contradictions(recs) == []


def test_wide_spread_is_a_value_disagreement():
    recs = [rec("a", "Summ", "XSum", "ROUGE-L", 25.0, ["BART"]),
            rec("b", "Summ", "XSum", "ROUGE-L", 45.0, ["BART"])]
    cons = find_contradictions(recs)
    kinds = {c.kind for c in cons}
    assert "value-disagreement" in kinds
    c = next(c for c in cons if c.kind == "value-disagreement")
    assert set(c.papers) == {"a", "b"} and c.spread > 0.15


def test_opposite_ranking_of_the_same_method_pair_is_flagged():
    recs = [
        rec("a", "Summ", "XSum", "ROUGE-L", 44.0, ["BART"]),
        {**rec("a2", "Summ", "XSum", "ROUGE-L", 40.0, ["PEGASUS"]), "paper_id": "a"},
        rec("b", "Summ", "XSum", "ROUGE-L", 41.0, ["BART"]),
        {**rec("b2", "Summ", "XSum", "ROUGE-L", 46.0, ["PEGASUS"]), "paper_id": "b"},
    ]
    kinds = {c.kind for c in find_contradictions(recs)}
    assert "ranking-disagreement" in kinds


def test_classifier_contrasts_edge_becomes_a_contradiction():
    cons = find_contradictions([], [{"citing_id": "a", "cited_id": "b",
                                     "relation": "contrasts", "prob": 0.9,
                                     "sentence": "unlike X we find Y"}])
    assert [c.kind for c in cons] == ["classifier-contrasts"]
    assert contradictions_table(cons)[0]["id"] == "C1"


def test_matrix_and_empty_cell_gap():
    recs = [rec("a", "Summ", "XSum", "ROUGE-L", 40, ["BART"], families=["Pretrained seq2seq"]),
            rec("b", "Summ", "XSum", "ROUGE-L", 41, ["BART"], families=["Pretrained seq2seq"]),
            rec("c", "Summ", "CNN/DailyMail", "ROUGE-L", 44, ["GPT-4"],
                families=["Large language model (prompted)"]),
            rec("d", "Summ", "CNN/DailyMail", "ROUGE-L", 43, ["GPT-4"],
                families=["Large language model (prompted)"])]
    m = build_matrix(recs)
    assert set(m["rows"]) == {"Pretrained seq2seq", "Large language model (prompted)"}
    assert set(m["columns"]) == {"XSum", "CNN/DailyMail"}
    gaps = candidate_gaps(m, [], recs)
    empties = {(g.method_family, g.dataset) for g in gaps if g.kind == "empty-cell"}
    assert ("Pretrained seq2seq", "CNN/DailyMail") in empties


def test_untested_comparison_requires_no_compares_against_edge():
    recs = [rec("a", "Summ", "XSum", "ROUGE-L", 40, ["BART"], families=["Pretrained seq2seq"]),
            rec("b", "Summ", "XSum", "ROUGE-L", 41, ["GPT-4"],
                families=["Large language model (prompted)"])]
    m = build_matrix(recs)
    assert any(g.kind == "untested-comparison" for g in candidate_gaps(m, [], recs))
    edges = [{"citing_id": "a", "cited_id": "b", "relation": "compares-against"}]
    assert not any(g.kind == "untested-comparison" for g in candidate_gaps(m, edges, recs))


def test_graph_has_typed_edges():
    recs = [rec("a", "Summ", "XSum", "ROUGE-L", 40, ["BART"]),
            rec("b", "Summ", "XSum", "ROUGE-L", 41, ["BART"])]
    g = build_graph(recs, [{"citing_id": "a", "cited_id": "b", "relation": "extends",
                            "prob": 0.8, "sentence": "we extend"}])
    assert g.number_of_nodes() == 2
    assert [d["relation"] for *_e, d in g.edges(data=True)] == ["extends"]
