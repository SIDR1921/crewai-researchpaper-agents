"""Unit tests for the LaTeX miner, including every parser failure found in the pilot."""
import pytest

from litreview.mining import latex_mining as lm


def doc(body: str, preamble: str = "") -> str:
    return r"\documentclass{article}" + preamble + r"\begin{document}" + body + r"\end{document}"


# ------------------------------------------------------------------ basic path

def test_minimal_paper_round_trip():
    res = lm.mine_paper({
        "main.tex": doc(r"\section{Related Work}"
                        r"Pointer networks were used by \citet{see2017}. "
                        r"In contrast, \citep{liu2019} fine-tune BERT encoders."
                        r"\section{Method}x"),
        "refs.bib": "@article{see2017,\n title = {Get to the point},\n}\n",
    })
    assert res["related_work"]["cited_keys"] == ["liu2019", "see2017"]
    assert res["keymap"]["see2017"] == "Get to the point"


def test_strip_comments_keeps_escaped_percent():
    assert lm.strip_comments(r"a % comment") == "a "
    assert lm.strip_comments(r"95\% of cases") == r"95\% of cases"


def test_citation_contexts_window_and_multi_key():
    text = ("First sentence about summarization models. "
            "Several studies [CITE:a,b] report gains on this task. "
            "Third sentence closes the paragraph here.")
    ctx = lm.citation_contexts(text)
    assert [c["cite_key"] for c in ctx] == ["a", "b"]
    assert ctx[0]["n_cites_in_sentence"] == 2
    assert "First sentence" in ctx[0]["context"] and "Third sentence" in ctx[0]["context"]


# --------------------------------------------- pilot failure case: main .tex choice

def test_multiple_documentclass_prefers_the_cited_paper():
    files = {
        "supplement.tex": doc(r"\section{Extra}no citations here at all"),
        "paper.tex": doc(r"\section{Related Work}Work by \citet{a} and \citet{b} and \citet{c}."),
    }
    assert lm.find_main_tex(files) == "paper.tex"


def test_main_tex_without_documentclass_falls_back_to_document_env():
    files = {"style.tex": r"\newcommand{\x}{y}",
             "body.tex": r"\begin{document}\section{Related Work}See \citet{a}.\end{document}"}
    assert lm.find_main_tex(files) == "body.tex"


def test_inline_inputs_resolves_subdirectories_and_bare_argument():
    files = {
        "main.tex": doc(r"\input{sections/related}\input sections/method"),
        "sections/related.tex": r"\section{Related Work}Prior work \citet{a} and \citet{b}.",
        "sections/method.tex": r"\section{Method}we do things",
    }
    tex = lm.inline_inputs(lm.strip_comments(files["main.tex"]), files)
    assert "Related Work" in tex and "Method" in tex


def test_inline_inputs_survives_self_reference():
    files = {"main.tex": doc(r"\input{loop}"), "loop.tex": r"\input{loop} tail"}
    assert "tail" in lm.inline_inputs(files["main.tex"], files)


# ------------------------------- pilot failure case: related work is not a \section

def test_related_work_as_subsection():
    res = lm.mine_paper({"main.tex": doc(
        r"\section{Background}\subsection{Related Work}"
        r"Earlier systems \citep{a} did X. Later ones \citep{b} did Y."
        r"\subsection{Notation}z")})
    rw = res["related_work"]
    assert rw["level"] == "subsection" and rw["cited_keys"] == ["a", "b"]


def test_related_work_only_as_paragraph_blocks():
    res = lm.mine_paper({"main.tex": doc(
        r"\section{Introduction}We introduce a system."
        r"\paragraph{Summarization.}Abstractive work \citep{a} and \citep{b} and \citep{c} exists."
        r"\paragraph{Contributions.}We do three things.")})
    assert res["related_work"] is not None
    assert set(res["related_work"]["cited_keys"]) == {"a", "b", "c"}


def test_background_heading_without_citations_is_not_related_work():
    res = lm.mine_paper({"main.tex": doc(
        r"\section{Background}We define notation and recall the transformer equations."
        r"\section{Method}Our approach.")})
    assert res["related_work"] is None


def test_explicit_related_work_beats_background():
    res = lm.mine_paper({"main.tex": doc(
        r"\section{Background}Notation \citep{x} and \citep{y}."
        r"\section{Related Work}Prior systems \citep{a} and \citep{b}.")})
    assert res["related_work"]["title"] == "Related Work"


def test_subsection_body_stops_at_next_section():
    secs = lm.extract_sections(
        doc(r"\section{A}\subsection{Related Work}rw body\section{B}b body"),
        levels=("subsection",))
    assert "b body" not in secs[0]["body"]


def test_starred_section_and_label_in_title():
    res = lm.mine_paper({"main.tex": doc(
        r"\section*{Related Work\label{sec:rw}}Prior \citep{a} and \citep{b}.")})
    assert res["related_work"]["title"] == "Related Work"


# ---------------------------------------- pilot failure case: bibliography formats

def test_parse_bbl_natbib_acl_style():
    bbl = r"""
\begin{thebibliography}{99}
\bibitem[{Lewis et~al.(2020)}]{lewis2020}
Mike Lewis, Yinhan Liu, and Naman Goyal. 2020.
\newblock BART: denoising sequence-to-sequence pre-training for generation.
\newblock In {\em Proceedings of ACL}, pages 7871--7880.
\end{thebibliography}"""
    assert "denoising sequence-to-sequence" in lm.parse_bbl(bbl)["lewis2020"]


def test_parse_bbl_biblatex_entry_format():
    bbl = r"""
\entry{smith2021}{article}{}
  \name{author}{1}{}{{hash=a}{family={Smith}}}
  \field{sortinit}{S}
  \field{title}{Faithfulness evaluation for {abstractive} summarization}
  \field{journaltitle}{TACL}
\endentry"""
    assert lm.parse_bbl(bbl)["smith2021"] == "Faithfulness evaluation for abstractive summarization"


def test_parse_bbl_title_first_when_no_author_block():
    bbl = r"""\bibitem{anon2022}
\newblock A study of citation intent classification.
\newblock In Proceedings of EMNLP."""
    assert lm.parse_bbl(bbl)["anon2022"] == "A study of citation intent classification"


def test_parse_bbl_skips_venue_block_starting_with_in():
    bbl = r"""\bibitem[{Doe(2019)}]{doe2019}
Jane Doe. 2019.
\newblock In Proceedings of NAACL.
\newblock Learning to summarise scientific articles.
"""
    assert lm.parse_bbl(bbl)["doe2019"] == "Learning to summarise scientific articles"


def test_parse_bib_nested_braces_and_multiline():
    bib = """@inproceedings{cohan2019,
  author = {Cohan, Arman},
  title  = {Structural {Scaffolds} for
            Citation Intent Classification},
  year   = {2019}
}
@article{other, title = "A quoted title", year = {2020}}
"""
    out = lm.parse_bib(bib)
    assert out["cohan2019"] == "Structural Scaffolds for Citation Intent Classification"
    assert out["other"] == "A quoted title"


def test_bbl_wins_over_bib_for_same_key():
    files = {"r.bib": "@article{k, title = {From bib},\n}",
             "r.bbl": r"\bibitem{k}" "\n" r"\newblock From bbl." "\n"}
    assert lm.build_keymap(files)["k"] == "From bbl"


def test_thebibliography_inlined_in_main_tex():
    main = doc(r"\section{Related Work}See \citet{k1} and \citet{k2}."
               r"\begin{thebibliography}{9}"
               r"\bibitem{k1}A. Author. \newblock An inlined title here. \newblock In ACL."
               r"\end{thebibliography}")
    res = lm.mine_paper({"main.tex": main})
    assert res["keymap"]["k1"] == "An inlined title here"


# ------------------------------------------------------- citation macro coverage

@pytest.mark.parametrize("macro", [
    r"\cite{a}", r"\citep{a}", r"\citet{a}", r"\citealp{a}", r"\citeA{a}",
    r"\parencite{a}", r"\autocite{a}", r"\textcite{a}", r"\shortcite{a}",
    r"\newcite{a}", r"\citep[see][p.~3]{a}", r"\citet*{a}", r"\Citep{a}",
])
def test_citation_macro_variants_detected(macro):
    assert lm.CITE_RE.findall(macro) == ["a"]


def test_latex_to_text_keeps_cite_markers_and_drops_floats():
    tex = (r"Results in \begin{table}\caption{drop me}\end{table} show gains \citep{a}. "
           r"See \ref{fig:1}.")
    text = lm.latex_to_text(tex)
    assert "[CITE:a]" in text and "drop me" not in text


# ------------------------------------------------------------------ section map

def test_section_map_buckets_canonical_names():
    sm = lm.section_map(doc(
        r"\section{Introduction}" + "Summarization matters a lot. " * 8 +
        r"\section{Experimental Results}" + "We report ROUGE scores on the benchmark. " * 8 +
        r"\section{Conclusion}" + "We conclude that the method helps a great deal. " * 8,
        preamble=r"") .replace(r"\begin{document}", r"\begin{document}\begin{abstract}" +
                               "We present a method for grounded review generation. " * 4 +
                               r"\end{abstract}"))
    assert {"abstract", "introduction", "results", "conclusion"} <= set(sm)
    assert "ROUGE" in sm["results"]
