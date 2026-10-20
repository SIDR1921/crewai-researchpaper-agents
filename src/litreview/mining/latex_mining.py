"""Core LaTeX mining logic: related-work sections, citation contexts, bibliography keys.

Pure functions, no network access, so they can be unit-tested offline.

Beyond the first working version this handles the failure modes seen in a real
arXiv pilot: ``biblatex``-style ``.bbl`` files, related work that is a
``\\subsection`` or only a ``\\paragraph``, several ``.tex`` files carrying
``\\documentclass``, ``\\thebibliography`` inlined in the main file, and the
wider family of citation macros used by ACL/IEEE/ACM styles.
"""
from __future__ import annotations

import re

from pylatexenc.latex2text import LatexNodes2Text

CITE_RE = re.compile(
    r"\\(?:"
    r"cite|citep|citet|citeA|citeNP|citealp|citealt|citeauthor|citeyear|citeyearpar"
    r"|parencite|textcite|autocite|smartcite|footcite|supercite|fullcite"
    r"|shortcite|newcite|nocite|citefullauthor|Citet|Citep|Autocite|Parencite|Textcite"
    r")\*?"
    r"(?:\[[^\]]*\]){0,2}\{([^}]+)\}"
)
SECTION_RE = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\s*\{")

#: Title patterns, most specific first. The first group that matches wins, so an
#: explicit "Related Work" is always preferred over a "Background" primer.
RW_TITLE_TIERS = [
    re.compile(r"related\s*(work|literature|studies|research)|prior\s*(work|art)|previous\s*work", re.I),
    re.compile(r"literature\s*review|related\s*approaches|existing\s*(work|approaches|methods)", re.I),
    re.compile(r"^background", re.I),
]
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}|\\(?:input|include)\s+([\w./\-]+)")

#: Canonical section buckets used by the Extractor (Stage 3).
SECTION_KINDS = {
    # Order matters: the first pattern that matches a heading wins, so the specific
    # buckets come before the broad ones ("Experimental Results" is results, not
    # experiments; "Model" only reaches `method` after everything else has had a look).
    "abstract": re.compile(r"^abstract", re.I),
    "introduction": re.compile(r"introduction|motivation", re.I),
    "related_work": RW_TITLE_TIERS[0],
    "limitations": re.compile(r"limitation|threat|ethic|broader\s*impact", re.I),
    "conclusion": re.compile(r"conclusion|conclud|future\s*work|summary", re.I),
    "results": re.compile(r"result|evaluation|analysis|finding|ablation|discussion", re.I),
    "data": re.compile(r"dataset|data\b|corpus|benchmark", re.I),
    "experiments": re.compile(r"experiment|setup|implementation|training", re.I),
    "method": re.compile(r"method|approach|model|architecture|our\s|framework|system", re.I),
}


def strip_comments(tex: str) -> str:
    """Remove LaTeX comments (unescaped %) line by line."""
    return re.sub(r"(?<!\\)%.*", "", tex)


def _braced(s: str, open_idx: int) -> tuple[str, int]:
    """Content of the brace group starting at ``open_idx`` ('{'), and the index after it.

    Brace-matching rather than ``[^}]+`` so nested groups -- common in titles and
    biblatex fields -- survive.
    """
    assert s[open_idx] == "{"
    depth, i = 0, open_idx
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1:i], i + 1
        i += 1
    return s[open_idx + 1:], len(s)


def find_main_tex(files: dict) -> str | None:
    """Pick the main ``.tex`` file.

    arXiv tarballs often contain several files with ``\\documentclass`` (a
    supplement, a rebuttal, a leftover template). Prefer one that also has
    ``\\begin{document}``, then the one with the most citation commands, which is
    the real paper rather than a one-page cover.
    """
    tex = {n: t for n, t in files.items() if n.lower().endswith(".tex")}
    cands = [n for n, t in tex.items() if "\\documentclass" in t]
    if not cands:
        # Some submissions ship a body file plus a style-only wrapper; fall back to
        # whichever .tex declares a document environment, then to the largest.
        cands = [n for n, t in tex.items() if "\\begin{document}" in t] or list(tex)
    if not cands:
        return None
    with_doc = [n for n in cands if "\\begin{document}" in tex[n]] or cands
    not_supp = [n for n in with_doc
                if not re.search(r"supp|appendi|rebuttal|response|cover|readme", n, re.I)] or with_doc

    def score(n: str) -> tuple[int, int, int]:
        t = tex[n]
        return (len(CITE_RE.findall(t)), len(INPUT_RE.findall(t)), len(t))

    return max(not_supp, key=score)


def inline_inputs(tex: str, files: dict, depth: int = 0, seen: frozenset = frozenset()) -> str:
    """Recursively replace ``\\input{x}`` / ``\\include{x}`` with file contents."""
    if depth > 6:
        return tex

    def repl(m):
        name = (m.group(1) or m.group(2) or "").strip().strip('"')
        name = re.sub(r"^\./", "", name)
        for cand in (name, name + ".tex"):
            for fname, content in files.items():
                norm = re.sub(r"^\./", "", fname)
                if norm == cand or norm.endswith("/" + cand) or norm.split("/")[-1] == cand:
                    if fname in seen:
                        return ""
                    return inline_inputs(strip_comments(content), files, depth + 1, seen | {fname})
        return ""

    return INPUT_RE.sub(repl, tex)


def document_body(tex: str) -> str:
    """Text between ``\\begin{document}`` and ``\\end{document}`` (or the whole string)."""
    body = tex.split("\\begin{document}", 1)[-1]
    return body.split("\\end{document}", 1)[0]


def extract_sections(tex: str, levels: tuple[str, ...] = ("section",)) -> list[dict]:
    """Split the body into sections at the requested heading levels.

    Each entry is ``{level, title, body}``; a section's body ends at the next
    heading of the same *or higher* level, so a ``\\subsection`` body stops at the
    next ``\\section``.
    """
    body = document_body(tex)
    order = {"section": 0, "subsection": 1, "subsubsection": 2, "paragraph": 3}
    marks = []
    for m in SECTION_RE.finditer(body):
        title, after = _braced(body, m.end() - 1)
        marks.append({"start": m.start(), "after": after, "level": m.group(1), "title": title.strip()})
    out = []
    for i, mk in enumerate(marks):
        if mk["level"] not in levels:
            continue
        end = len(body)
        for nxt in marks[i + 1:]:
            if order[nxt["level"]] <= order[mk["level"]]:
                end = nxt["start"]
                break
        out.append({"level": mk["level"], "title": clean_title(mk["title"]), "body": body[mk["after"]:end]})
    return out


def clean_title(title: str) -> str:
    title = re.sub(r"\\label\s*\{[^}]*\}", "", title)
    title = re.sub(r"\\[a-zA-Z]+\*?", "", title)
    return re.sub(r"[{}$]|\s+", lambda m: " " if m.group().isspace() else "", title).strip()


def find_related_work(tex: str) -> dict | None:
    """Best related-work block, searching sections, subsections and paragraphs.

    Tiered so an explicit "Related Work" beats "Literature Review" which beats a
    bare "Background". Within a tier, a ``\\section`` beats a ``\\subsection``, and
    a longer block beats a shorter one.
    """
    secs = extract_sections(tex, levels=("section", "subsection", "subsubsection", "paragraph"))
    rank = {"section": 0, "subsection": 1, "subsubsection": 2, "paragraph": 3}
    for tier, pat in enumerate(RW_TITLE_TIERS):
        hits = [s for s in secs if pat.search(s["title"])]
        if not hits:
            continue
        best = min(hits, key=lambda s: (rank[s["level"]], -len(s["body"])))
        # A heading alone proves nothing: an explicit "Related Work" needs at least one
        # citation, and a looser heading such as "Background" needs at least two before we
        # treat it as a related-work section rather than a technical primer.
        if len(CITE_RE.findall(best["body"])) < (1 if tier == 0 else 2):
            continue
        return {**best, "tier": tier}
    # Papers with no related-work heading at all sometimes group it under
    # consecutive \paragraph blocks that are dense in citations.
    paras = [s for s in secs if s["level"] == "paragraph" and len(CITE_RE.findall(s["body"])) >= 3]
    if paras:
        best = max(paras, key=lambda s: len(CITE_RE.findall(s["body"])))
        return {**best, "tier": len(RW_TITLE_TIERS)}
    return None


def section_map(tex: str) -> dict[str, str]:
    """Canonical-name -> plain text, for the sections the Extractor needs."""
    out: dict[str, str] = {}
    for sec in extract_sections(tex, levels=("section", "subsection")):
        for kind, pat in SECTION_KINDS.items():
            if pat.search(sec["title"]):
                text = latex_to_text(sec["body"])
                if len(text) > 120:
                    out[kind] = (out.get(kind, "") + "\n\n" + text).strip()
                break
    if m := re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S):
        out["abstract"] = latex_to_text(m.group(1))
    return out


def _cite_to_placeholder(tex: str) -> str:
    """Replace citation commands with tokens that survive LaTeX->text conversion."""
    return CITE_RE.sub(lambda m: " CITEOPEN" + m.group(1).replace(" ", "") + "CITECLOSE ", tex)


_l2t = LatexNodes2Text(math_mode="verbatim", keep_comments=False)


def latex_to_text(tex: str) -> str:
    """Convert LaTeX to plain text while keeping citation markers as ``[CITE:keys]``."""
    tex = re.sub(
        r"\\begin\{(figure|table|tabular|algorithm|algorithmic|equation|align|lstlisting|verbatim|wrapfigure)\*?\}"
        r".*?\\end\{\1\*?\}", " ", tex, flags=re.S)
    tex = re.sub(r"\\(?:label|ref|autoref|eqref|cref|Cref|footnote|url|href)\s*\{[^}]*\}", " ", tex)
    tex = _cite_to_placeholder(tex)
    try:
        text = _l2t.latex_to_text(tex)
    except Exception:
        text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", tex)
    text = re.sub(r"CITEOPEN(.*?)CITECLOSE", lambda m: f"[CITE:{m.group(1)}]", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([.,;:)])", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
_ABBREV = re.compile(r"\b(e\.g|i\.e|et al|cf|vs|resp|approx|Fig|Figs|Sec|Secs|Eq|Eqs|Tab|Ref|Refs|al|No|vol)\.")


def split_sentences(text: str) -> list[str]:
    text = _ABBREV.sub(lambda m: m.group(1) + "<DOT>", text)
    sents = [s.replace("<DOT>", ".").strip() for s in SENT_SPLIT.split(text)]
    return [s for s in sents if len(s) > 20]


def citation_contexts(text: str, window: int = 1) -> list[dict]:
    """For every sentence with a ``[CITE:...]`` marker, emit one record per cited key.

    ``context`` includes ``window`` sentences either side; ``cite_sentence`` is the
    sentence itself.
    """
    sents = split_sentences(text)
    out = []
    for i, s in enumerate(sents):
        keys = re.findall(r"\[CITE:([^\]]+)\]", s)
        if not keys:
            continue
        ctx = " ".join(sents[max(0, i - window): i + window + 1])
        all_keys = [k for group in keys for k in group.split(",") if k]
        for k in all_keys:
            out.append({
                "cite_key": k,
                "cite_sentence": s,
                "context": ctx,
                "n_cites_in_sentence": len(all_keys),
            })
    return out


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 80]


# ---------- bibliography ----------

_AUTHORY = re.compile(r"[A-Z]\.|\band\b|~")


def _clean_title(raw: str) -> str:
    t = re.sub(r"\\(?:emph|textit|textbf|em|bf|it|mbox|text)\s*\{", "{", raw)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}]", "", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\b(doi|arXiv|URL|https?://\S+)\b.*$", "", t, flags=re.I)
    return t.strip(" .,;:")


def _best_title(blocks: list[str]) -> str:
    """Of a ``\\bibitem``'s ``\\newblock`` blocks, the one that looks like a title.

    Styles disagree on block order, so score instead of indexing: penalise
    author-looking blocks and venue blocks that start with "In".
    """
    best, best_score = "", -1e9
    for i, b in enumerate(blocks):
        t = _clean_title(b)
        if len(t) < 8:
            continue
        score = 0.0
        score -= 40 if re.match(r"^in\b", t, re.I) else 0
        score -= 25 * len(_AUTHORY.findall(b[:120])) / 3
        score -= 30 if re.match(r"^(proceedings|journal|transactions|advances|arxiv|acm|ieee)", t, re.I) else 0
        score += 20 if 20 <= len(t) <= 220 else 0
        score -= abs(len(t) - 80) / 20
        score += 8 if i == 1 else 0        # most styles put the title second
        score -= 20 if re.search(r"\b(19|20)\d\d\b", t) and len(t) < 40 else 0
        if score > best_score:
            best, best_score = t, score
    return best


def parse_bbl(bbl: str) -> dict:
    """Map cite key -> title from a compiled ``.bbl``.

    Handles both the ``\\bibitem``/``\\newblock`` families (plain, unsrt, natbib,
    acl) and ``biblatex``'s ``\\entry{key}{type}{}`` + ``\\field{title}{...}``.
    """
    out: dict[str, str] = {}

    # biblatex
    for m in re.finditer(r"\\entry\{([^}]+)\}\{[^}]*\}", bbl):
        key, start = m.group(1).strip(), m.end()
        end = bbl.find("\\endentry", start)
        body = bbl[start: end if end != -1 else len(bbl)]
        title = ""
        for fm in re.finditer(r"\\field\s*\{(title|maintitle|booktitle|labeltitle)\}\s*\{", body):
            raw, _ = _braced(body, fm.end() - 1)
            if fm.group(1) in ("title", "labeltitle"):
                title = raw
                break
            title = title or raw
        if title:
            out[key] = _clean_title(title)

    # \bibitem families
    for m in re.finditer(
        r"\\bibitem(?:\[[^\]]*(?:\{[^}]*\}[^\]]*)*\])?\s*\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)",
        bbl, re.S,
    ):
        key, entry = m.group(1).strip(), m.group(2)
        blocks = [b.strip() for b in re.split(r"\\newblock", entry) if b.strip()]
        title = _best_title(blocks) if len(blocks) > 1 else _clean_title(entry)
        if title:
            out.setdefault(key, title)
    return out


def parse_bib(bib: str) -> dict:
    """Map cite key -> title from a ``.bib`` file (brace-matched, multi-line safe)."""
    out: dict[str, str] = {}
    for m in re.finditer(r"@\w+\s*[{(]\s*([^,\s]+)\s*,", bib):
        key, start = m.group(1), m.end()
        nxt = bib.find("\n@", start)
        body = bib[start: nxt if nxt != -1 else len(bib)]
        tm = re.search(r"\btitle\s*=\s*", body, re.I)
        if not tm:
            continue
        rest = body[tm.end():].lstrip()
        if rest.startswith("{"):
            raw, _ = _braced(rest, 0)
        elif rest.startswith('"'):
            raw = rest[1:rest.find('"', 1)] if rest.find('"', 1) != -1 else rest[1:]
        else:
            raw = rest.split(",")[0]
        title = _clean_title(raw)
        if title:
            out[key] = title
    return out


def build_keymap(files: dict, inlined_tex: str | None = None) -> dict:
    """Cite key -> title, from ``.bib``, then ``.bbl`` (which wins), then any
    ``thebibliography`` block inlined in the main ``.tex``."""
    keymap: dict[str, str] = {}
    for name, content in files.items():
        if name.lower().endswith(".bib"):
            keymap.update(parse_bib(content))
    for name, content in files.items():  # .bbl wins: it's what was actually compiled
        if name.lower().endswith(".bbl"):
            keymap.update(parse_bbl(content))
    if inlined_tex and "\\bibitem" in inlined_tex:
        for k, v in parse_bbl(inlined_tex).items():
            keymap.setdefault(k, v)
    return keymap


def mine_paper(files: dict) -> dict | None:
    """``files``: ``{relative_path: text}``. Related work, contexts, sections, keymap."""
    main = find_main_tex(files)
    if not main:
        return None
    tex = inline_inputs(strip_comments(files[main]), files)
    rw = find_related_work(tex)
    keymap = build_keymap(files, inlined_tex=tex)
    full_text = latex_to_text(document_body(tex))
    result = {
        "main_file": main,
        "keymap": keymap,
        "all_contexts": citation_contexts(full_text),
        "sections": section_map(tex),
        "n_cites_total": len(CITE_RE.findall(tex)),
        "related_work": None,
    }
    if rw:
        rw_text = latex_to_text(rw["body"])
        result["related_work"] = {
            "title": rw["title"],
            "level": rw["level"],
            "tier": rw["tier"],
            "text": rw_text,
            "paragraphs": paragraphs(rw_text),
            "cited_keys": sorted({c["cite_key"] for c in citation_contexts(rw_text)}),
        }
    return result
