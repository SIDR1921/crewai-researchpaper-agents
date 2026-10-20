"""Demo UI for the literature-review assistant.

    streamlit run app/streamlit_app.py

Deliberately thin: it drives :class:`litreview.flow.LitReviewFlow` one step at a
time so the human approval checkpoint is a real button rather than a formality,
and shows what each stage produced. It adds no logic of its own -- anything you
can see here came from the package, and everything is cached in the run
directory, so closing the tab does not lose work.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from litreview.config import SETTINGS, paths
from litreview.flow import LitReviewFlow
from litreview.llm import install_call_logger, probe, usage_summary

st.set_page_config(page_title="Literature review assistant", layout="wide")
P = paths().mkdirs()
install_call_logger(P.results / "llm_calls.jsonl")

STEPS = ["collect_corpus", "approve", "extract", "verify_claims", "classify_relations",
         "build_map", "find_gaps", "write_review", "verify_review", "export"]


def flow() -> LitReviewFlow:
    return st.session_state["flow"]


with st.sidebar:
    st.header("Configuration")
    st.caption("One config drives every LLM call; see `litreview/config.py`.")
    st.json(SETTINGS.describe(), expanded=False)
    if st.button("Probe provider"):
        st.json(probe())
    st.divider()
    topic = st.text_area(
        "Topic", "faithfulness and hallucination evaluation in abstractive summarisation",
        height=80)
    n_papers = st.slider("Corpus size", 5, 40, 12)
    seeds = st.text_area("Seed paper titles (one per line, optional)", "", height=70)
    use_verifier = st.checkbox("Use the Verifier", True)
    use_relations = st.checkbox("Use the relation classifier", True)
    if st.button("Start / reset run", type="primary"):
        clf = None
        if use_relations and (P.models / "scibert_relations").exists():
            from litreview.relations.scibert import RelationClassifier
            clf = RelationClassifier(P.models / "scibert_relations")
        elif use_relations:
            st.warning("No classifier at models/scibert_relations; relations will be skipped.")
        st.session_state["flow"] = LitReviewFlow(
            topic=topic, n_papers=n_papers,
            seeds=[s.strip() for s in seeds.splitlines() if s.strip()],
            approve=lambda t: st.session_state.get("approved", False),
            classifier=clf, use_relations=use_relations, use_verifier=use_verifier)
        st.session_state["done"] = []
        st.session_state["approved"] = False
        st.rerun()

st.title("Grounded literature review assistant")

if "flow" not in st.session_state:
    st.info("Set a topic in the sidebar and press **Start / reset run**.")
    st.stop()

f = flow()
done = st.session_state["done"]
st.caption(f"Run directory: `{f.state.run_dir}`")
st.progress(len(done) / len(STEPS), text=f"{len(done)}/{len(STEPS)} steps complete")

tabs = st.tabs(["Corpus", "Claims", "Map", "Gaps", "Review", "Usage"])

# ---------------------------------------------------------------- 1-2. corpus
with tabs[0]:
    if "collect_corpus" not in done:
        if st.button("1. Collect corpus"):
            with st.spinner("searching OpenAlex and snowballing references..."):
                f.collect_corpus()
            done.append("collect_corpus")
            st.rerun()
    else:
        from litreview.corpus import corpus_table
        st.json(f.state.corpus_stats, expanded=False)
        st.dataframe(pd.DataFrame(corpus_table(f.state.corpus)), use_container_width=True)
        if "approve" not in done:
            st.warning("Human approval checkpoint: the pipeline stops here until you decide.")
            c1, c2 = st.columns(2)
            if c1.button("Approve corpus", type="primary"):
                st.session_state["approved"] = True
                f.approve_corpus()
                done.append("approve")
                st.rerun()
            if c2.button("Reject"):
                st.session_state["approved"] = False
                f.approve_corpus()
                st.error("Corpus rejected. Change the topic or the size and start again.")
        else:
            st.success("Corpus approved.")

# ------------------------------------------------------------- 3-4. extract
with tabs[1]:
    if "approve" not in done:
        st.info("Approve a corpus first.")
    elif "extract" not in done:
        if st.button("3. Extract claims"):
            with st.spinner("extracting (one LLM call per paper, cached)..."):
                f.extract()
            done.append("extract")
            st.rerun()
    else:
        st.metric("papers extracted", len(f.state.records))
        st.metric("claims", sum(len(r.get("claims") or []) for r in f.state.records))
        if "verify_claims" not in done:
            if st.button("4. Verify claims", type="primary"):
                with st.spinner("quote matching + NLI entailment..."):
                    f.verify_claims()
                done.append("verify_claims")
                st.rerun()
        else:
            st.json(f.state.verify_report, expanded=False)
        rows = []
        for r in f.state.records:
            vs = {v["claim_id"]: v for v in (f.state.claim_verdicts.get(r["paper_id"]) or [])}
            for c in (r.get("claims") or []):
                v = vs.get(c["claim_id"], {})
                rows.append({"paper": r["paper_id"], "claim": c["text"][:110],
                             "type": c["type"], "verdict": v.get("verdict", "-"),
                             "quote": v.get("quote_score"), "entail": v.get("entail_prob"),
                             "evidence": c["evidence"]["passage"][:140]})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)

# ------------------------------------------------------------- 5-6. the map
with tabs[2]:
    if "verify_claims" not in done and "extract" not in done:
        st.info("Extract first.")
    else:
        if "build_map" not in done:
            if st.button("5-6. Classify relations and build the map"):
                with st.spinner("classifying relations, grouping results, adjudicating..."):
                    f.classify_relations()
                    f.build_map()
                done += ["classify_relations", "build_map"]
                st.rerun()
        else:
            st.subheader("Consensus groups")
            g = pd.DataFrame(f.state.consensus.get("groups") or [])
            if len(g):
                st.dataframe(g.drop(columns=["values", "papers"], errors="ignore"),
                             use_container_width=True)
            st.subheader("Contradictions")
            c = pd.DataFrame(f.state.contradictions)
            st.dataframe(c, use_container_width=True) if len(c) else st.caption("none flagged")
            gp = Path(f.state.run_dir) / "citation_graph.html"
            if gp.exists():
                st.subheader("Typed citation graph")
                st.components.v1.html(gp.read_text(), height=640, scrolling=True)

# ---------------------------------------------------------------- 7. gaps
with tabs[3]:
    if "build_map" not in done:
        st.info("Build the map first.")
    elif "find_gaps" not in done:
        if st.button("7. Find gaps"):
            with st.spinner("building the coverage matrix and arguing each candidate..."):
                f.find_gaps()
            done.append("find_gaps")
            st.rerun()
    else:
        m = f.state.matrix
        if m.get("rows"):
            st.subheader("Method family x dataset coverage")
            st.dataframe(pd.DataFrame(m["matrix"], index=m["rows"], columns=m["columns"]),
                         use_container_width=True)
        st.subheader("Candidate gaps, with the Analyst's argument both ways")
        gp = pd.DataFrame(f.state.gaps)
        if len(gp):
            st.dataframe(gp[["kind", "method_family", "dataset", "verdict", "score",
                             "candidate", "argument_for", "argument_against"]],
                         use_container_width=True, height=400)

# ------------------------------------------------------------- 8-10. review
with tabs[4]:
    if "find_gaps" not in done:
        st.info("Find gaps first.")
    elif "export" not in done:
        if st.button("8-10. Write, verify and export", type="primary"):
            with st.spinner("writing, verifying every sentence, revising once, exporting..."):
                f.write_review()
                f.verify_review()
                f.export()
            done += ["write_review", "verify_review", "export"]
            st.rerun()
    else:
        fa = f.state.faithfulness or {}
        if "after_revision" in fa:
            c1, c2, c3 = st.columns(3)
            c1.metric("faithfulness", f"{fa['after_revision']['faithfulness']:.0%}")
            c2.metric("before revision", f"{fa['before_revision']['faithfulness']:.0%}")
            c3.metric("still flagged", fa.get("n_flagged_after", 0))
        st.markdown(f.state.review.get("markdown", ""))
        with st.expander("Sentence-level verdicts"):
            sv = pd.DataFrame(fa.get("sentence_verdicts") or [])
            st.dataframe(sv, use_container_width=True) if len(sv) else st.caption("none")
        st.subheader("Exports")
        for k, v in (f.state.exports or {}).items():
            p = Path(v)
            if p.exists():
                st.download_button(f"Download {k} ({p.name})", p.read_bytes(), file_name=p.name,
                                   key=f"dl_{k}")

# ---------------------------------------------------------------- usage
with tabs[5]:
    u = usage_summary()
    o = u["overall"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM calls", o["calls"])
    c2.metric("tokens", f"{o['total_tokens']:,}")
    c3.metric("cost (USD)", f"{o['cost_usd']:.4f}")
    c4.metric("mean latency", f"{o['latency_s_mean'] or 0:.1f}s")
    st.caption("Every call in the project is logged by a LiteLLM callback; this is the same "
               "file the report's cost table is built from.")
    st.dataframe(pd.DataFrame(u["by_tag"]).T, use_container_width=True)
