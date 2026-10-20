/* Literature Review console.
   Thin on purpose: it fetches, it renders, it starts jobs. Every number shown
   comes from the API, which reads the pipeline's own result files. */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const RELATION_COLOUR = {
  "background": "#6d80a5", "uses-method": "#4f8ed6", "uses-data": "#4fa97a",
  "extends": "#9b74c9", "compares-against": "#e1883a", "supports": "#5fd4c4",
  "contrasts": "#d6604d",
};

const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nf = n => (n === null || n === undefined || n === "") ? "—"
  : (typeof n === "number" ? n.toLocaleString() : String(n));

async function api(path) {
  try {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${r.status}`);
    return await r.json();
  } catch (e) { return { error: String(e.message || e) }; }
}

/* ───────────────────────────── navigation ───────────────────────────── */
const LOADERS = {};
function go(view) {
  $$(".view").forEach(v => v.classList.toggle("is-active", v.dataset.view === view));
  $$(".nav-item").forEach(b => b.classList.toggle("is-active", b.dataset.go === view));
  document.documentElement.dataset.view = view;
  history.replaceState(null, "", "#" + view);
  LOADERS[view]?.();
}
document.addEventListener("click", e => {
  const b = e.target.closest("[data-go]");
  if (b) { e.preventDefault(); go(b.dataset.go); }
});

/* ───────────────────────────── overview ─────────────────────────────── */
LOADERS.overview = async () => {
  const st = await api("/api/status");
  if (st.error) { $("#stages").innerHTML = `<p class="err">${esc(st.error)}</p>`; return; }

  $("#stage-progress").textContent = `${st.done}/${st.total}`;
  const p = st.provider || {};
  $("#provider-text").textContent = `${p.provider ?? "?"} · ${(p.model ?? "").split("/").pop()}`;
  $("#provider").classList.add("ok");

  $("#stages").innerHTML = st.stages.map(s => `
    <li class="stage-card ${s.done ? "done" : ""}">
      <div class="no">Stage ${esc(s.n)}<span class="pill">${s.done ? "done" : "pending"}</span></div>
      <h4>${esc(s.name)}</h4>
      <div class="blurb">${esc(s.blurb)}</div>
      ${s.headline?.length ? `<div class="nums">${s.headline.map(h =>
        `<div><b>${nf(h.value)}</b>${esc(h.label)}</div>`).join("")}</div>` : ""}
    </li>`).join("");

  const d = await api("/api/dataset");
  const y = d.yield || {}, ds = (y.datasets || {}), yy = (y.yield || {});
  $("#overview-stats").innerHTML = [
    ["Benchmark examples", d.rw_pairs, "papers with a hidden related-work section", true],
    ["Citation contexts", d.contexts, "one per citing sentence × cited paper"],
    ["Labelled contexts", d.labels, "with a relation and a rationale"],
    ["Multi-XScience", d.multixscience, "external benchmark, same schema"],
  ].map(([k, v, note, accent]) => `
    <div class="stat ${accent ? "accent" : ""}">
      <div class="k">${esc(k)}</div><div class="v">${nf(v)}</div>
      <div class="note">${esc(note)}</div>
    </div>`).join("");

  const listed = yy.listed || 0;
  if (listed) {
    const steps = [
      ["listed", listed], ["LaTeX source", listed - (yy.no_source || 0)],
      ["parsed", yy.parsed], ["related work found", yy.with_related_work],
      ["kept as benchmark", ds.rw_pairs_kept],
    ];
    $("#yield-bars").innerHTML = steps.map(([lab, v]) => `
      <div class="bar-row">
        <div class="lab">${esc(lab)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${(100 * v / listed).toFixed(1)}%"></div></div>
        <div class="val">${nf(v)} · ${(100 * v / listed).toFixed(0)}%</div>
      </div>`).join("");
  } else {
    $("#yield-bars").innerHTML = `<p class="panel-sub">Stage 1 has not run yet.</p>`;
  }
};

/* ───────────────────────────── benchmark ────────────────────────────── */
let SPLIT = "";
LOADERS.benchmark = async () => {
  const d = await api(`/api/benchmark?limit=200&split=${SPLIT}`);
  const tb = $("#benchmark-table tbody");
  if (d.error || !d.items?.length) {
    tb.innerHTML = `<tr><td colspan="7" class="panel-sub">No benchmark examples yet — run Stage 1.</td></tr>`;
    return;
  }
  tb.innerHTML = d.items.map(r => `
    <tr data-id="${esc(r.arxiv_id)}">
      <td><span class="ttl">${esc(r.title || r.arxiv_id)}</span></td>
      <td><span class="tag ${esc(r.split)}">${esc(r.split)}</span></td>
      <td>${esc((r.published || "—").replace(/-/g, "\u2011"))}</td>
      <td>${esc(r.rw_title || "—")}<span class="sec">${esc(r.rw_level)}</span></td>
      <td class="num">${nf(r.words)}</td>
      <td class="num">${nf(r.n_cited)}</td>
      <td class="num">${r.coverage != null ? (100 * r.coverage).toFixed(0) + "%" : "—"}</td>
    </tr>`).join("");
};
$("#split-filter").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#split-filter button").forEach(x => x.classList.toggle("is-active", x === b));
  SPLIT = b.dataset.split; LOADERS.benchmark();
});
$("#benchmark-table").addEventListener("click", async e => {
  const tr = e.target.closest("tr[data-id]"); if (!tr) return;
  $$("#benchmark-table tbody tr").forEach(x => x.classList.toggle("is-open", x === tr));
  const r = await api(`/api/example?id=${encodeURIComponent(tr.dataset.id)}`);
  const box = $("#example-detail");
  if (r.error) { box.hidden = true; return; }
  const withAbs = (r.cited || []).filter(c => c.abstract).length;
  box.hidden = false;
  box.innerHTML = `
    <h3>${esc(r.title || r.arxiv_id)}</h3>
    <p class="panel-sub">arXiv ${esc(r.arxiv_id)} · ${esc(r.published)} ·
       ${nf((r.cited || []).length)} cited, ${nf(withAbs)} with an abstract</p>
    <h4 style="margin-top:15px;font-size:12px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-3)">
      The human section a system must regenerate</h4>
    <div class="rw">${esc(r.rw_text || "").replace(/\n/g, "<br>")}</div>
    <h4 style="margin-top:15px;font-size:12px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-3)">
      Papers it cites — the only input a system gets</h4>
    <div class="cited">${(r.cited || []).map(c => `
      <div class="c ${c.abstract ? "" : "no-abs"}">
        <b>${esc(c.title || c.cite_key)}</b>
        <span>${c.year ? esc(c.year) + " · " : ""}${c.abstract ? "abstract available" : "no abstract"}</span>
      </div>`).join("")}</div>`;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

/* ───────────────────────────── relations ────────────────────────────── */
LOADERS.relations = async () => {
  const d = await api("/api/labels");
  if (d.error || !d.n_labelled) {
    $("#relation-cards").innerHTML = `<p class="panel-sub">No labels yet — run Stage 2.</p>`;
    return;
  }
  $("#label-count").textContent = nf(d.n_labelled);
  const max = Math.max(...d.labels.map(l => l.count), 1);
  $("#relation-cards").innerHTML = d.labels.map(l => {
    const c = RELATION_COLOUR[l.label] || "#6d80a5";
    const ex = l.examples?.[0];
    return `<div class="rel ${l.count ? "" : "empty-class"}" style="--c:${c}">
      <div class="rel-head">
        <code>${esc(l.label)}</code>
        <span class="n">${nf(l.count)}</span>
        <span class="pc">${(100 * l.share).toFixed(1)}%</span>
      </div>
      <div class="track"><i style="width:${(100 * l.count / max).toFixed(1)}%"></i></div>
      <div class="def">${esc(l.definition)}</div>
      ${ex ? `<div class="ex">${esc(ex.context).replace(/\[\[CITED\]\]/g, "<mark>[[CITED]]</mark>")}
        ${ex.rationale ? `<span class="why">↳ ${esc(ex.rationale)}</span>` : ""}</div>`
        : `<div class="warn">The labeller never assigned this class. Whether that is
            genuine rarity or a blind spot is what the gold set decides.</div>`}
    </div>`;
  }).join("");
};

/* ──────────────────────────── grounded review ───────────────────────── */
LOADERS.review = async () => {
  const d = await api("/api/review");
  const body = $("#review-body"), ev = $("#evidence");
  if (d.empty || d.error || !d.sentences?.length) {
    $("#review-extras").hidden = true;
    body.innerHTML = `<div class="empty"><h3>No review yet</h3>
      <p>Run the Flow on a topic and it appears here, sentence by sentence,
         each one carrying the evidence it rests on.</p>
      <button class="btn" data-go="run">Go to Run →</button></div>`;
    return;
  }
  const f = d.faithfulness || {};
  $("#faith-chip").innerHTML = `<span>${f.faithfulness != null
    ? (100 * f.faithfulness).toFixed(0) + "%" : "—"}</span> of sentences supported`;

  // the human section this run is being measured against, when it is a demo
  const ref = d.reference;
  const human = $("#human-body");
  if (ref?.rw_text) {
    human.innerHTML = `<span class="panel-label">Written by the author of
        ${esc(ref.arxiv_id)}</span>` +
      esc(ref.rw_text).replace(/\[CITE:([^\]]+)\]/g,
        (_, k) => `<span class="cite">[${esc(k)}]</span>`).replace(/\n/g, "<br>");
  } else {
    human.innerHTML = `<span class="panel-label">No human reference</span>
      <p class="panel-sub">This run came from a topic search rather than a benchmark
         paper, so there is nothing to compare against.</p>`;
  }

  body.innerHTML = `<span class="panel-label">Generated by the crew</span>
    <div class="legend">
      <span><i style="background:#4fa97a"></i>supported</span>
      <span><i style="background:#e1883a"></i>partially</span>
      <span><i style="background:#d6604d"></i>unsupported</span>
      <span style="margin-left:auto">${esc(d.topic || "")}</span>
    </div>` + d.sentences.map((s, i) =>
      `<span class="sent ${esc(s.verdict)}" data-i="${i}">${esc(s.text)}</span> `).join("");

  body.addEventListener("click", e => {
    const el = e.target.closest(".sent"); if (!el) return;
    $$(".sent", body).forEach(x => x.classList.toggle("sel", x === el));
    const s = d.sentences[+el.dataset.i];
    const papers = (s.cited_paper_ids || []).map(p =>
      `<span class="tag">${esc((d.papers?.[p]?.title || p).slice(0, 44))}</span>`).join(" ");
    ev.innerHTML = `<h3>Evidence</h3>
      <p class="panel-sub">What this sentence was checked against.</p>
      <div class="ev-block">
        <div class="meta">
          <span class="verdict ${esc(s.verdict)}">${esc(s.verdict)}</span>
          <span>entailment ${s.entail_prob != null ? s.entail_prob.toFixed(3) : "—"}</span>
        </div>
        <q>${esc(s.text)}</q>
      </div>
      ${papers ? `<div class="ev-block"><div class="meta">cited papers</div>${papers}</div>` : ""}
      <div class="ev-block">
        <div class="meta">closest supporting passage</div>
        ${s.best_evidence ? `<q>${esc(s.best_evidence)}</q>`
          : `<p class="panel-sub">The Verifier found no passage that supports this
             sentence. That is why it is flagged.</p>`}
      </div>`;
  }, { once: false });

  $("#review-extras").hidden = false;
  $("#contradictions").innerHTML = (d.contradictions || []).length
    ? d.contradictions.slice(0, 8).map(c => `<div class="item">
        <b>${esc(c.kind)}</b> — ${esc(c.metric || "")} ${c.dataset ? "on " + esc(c.dataset) : ""}
        <span class="adj">${esc(c.adjudication || "not adjudicated")}</span></div>`).join("")
    : `<p class="panel-sub">None flagged in this run.</p>`;
  $("#gaps").innerHTML = (d.gaps || []).length
    ? d.gaps.slice(0, 8).map(g => `<div class="item">
        <b>${esc(g.candidate)}</b>
        <span class="verdict ${g.verdict === "real-gap" ? "supported" : "partially"}">${esc(g.verdict || "—")}</span>
        <span class="adj">${esc(g.argument_against || "")}</span></div>`).join("")
    : `<p class="panel-sub">No candidates in this run.</p>`;
};

$("#review-mode").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#review-mode button").forEach(x => x.classList.toggle("is-active", x === b));
  const compare = b.dataset.mode === "compare";
  $(".review-layout").classList.toggle("compare", compare);
  $("#human-body").hidden = !compare;
});

/* ─────────────────────────────── cost ───────────────────────────────── */
LOADERS.cost = async () => {
  const u = await api("/api/usage");
  if (u.error) return;
  const o = u.overall || {};
  $("#usage-stats").innerHTML = [
    ["Model calls", o.calls, `${nf(o.failed)} failed`, true],
    ["Tokens", o.total_tokens, `${nf(o.prompt_tokens)} in · ${nf(o.completion_tokens)} out`],
    ["Cost (USD)", o.cost_usd != null ? o.cost_usd.toFixed(4) : "—", "free tier bills at zero"],
    ["Mean latency", o.latency_s_mean != null ? o.latency_s_mean.toFixed(2) + "s" : "—",
     `${nf(o.latency_s_total)}s total`],
  ].map(([k, v, note, a]) => `<div class="stat ${a ? "accent" : ""}">
      <div class="k">${esc(k)}</div><div class="v">${nf(v)}</div>
      <div class="note">${esc(note)}</div></div>`).join("");

  const tags = Object.entries(u.by_tag || {})
    .filter(([, v]) => v.total_tokens).sort((a, b) => b[1].total_tokens - a[1].total_tokens);
  const max = Math.max(...tags.map(([, v]) => v.total_tokens), 1);
  $("#usage-bars").innerHTML = tags.length ? tags.map(([k, v]) => `
    <div class="bar-row"><div class="lab">${esc(k)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(100 * v.total_tokens / max).toFixed(1)}%"></div></div>
      <div class="val">${nf(v.total_tokens)} tok</div></div>`).join("")
    : `<p class="panel-sub">No calls logged yet.</p>`;
};

/* ──────────────────────────────── run ───────────────────────────────── */
let POLL = null;
LOADERS.run = async () => {
  if ($("#commands").dataset.ready) return;
  const d = await api("/api/commands");
  if (d.error) return;
  $("#commands").innerHTML = `<h3>Pipeline commands</h3>
    <p class="panel-sub">These start the project's own scripts. Nothing else can be run.
      On a free API tier each model call waits for the per-minute token budget, so a
      demo takes several minutes and the Flow longer. Progress appears on the right;
      everything is cached, so stopping and restarting loses nothing.</p>` +
    d.commands.map(c => `<div class="cmd" data-cmd="${esc(c.id)}">
      <h4>${esc(c.label)}</h4>
      <div class="fields">
        ${c.fields.map(f => f.type === "number"
          ? `<input type="number" data-f="${esc(f.name)}" value="${esc(f.value)}" min="1">`
          : `<input type="text" data-f="${esc(f.name)}" placeholder="${esc(f.placeholder || "")}">`
        ).join("")}
        <button class="btn" data-start="${esc(c.id)}">Start</button>
      </div></div>`).join("");
  $("#commands").dataset.ready = "1";
};

document.addEventListener("click", async e => {
  const b = e.target.closest("[data-start]"); if (!b) return;
  const card = b.closest(".cmd");
  const args = {};
  $$("input[data-f]", card).forEach(i => args[i.dataset.f] = i.value);
  b.disabled = true;
  const r = await fetch("/api/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd: b.dataset.start, args }),
  }).then(x => x.json()).catch(x => ({ error: String(x) }));
  b.disabled = false;
  if (r.error) { $("#log").textContent = "Could not start: " + r.error; return; }
  $("#log").textContent = "starting…";
  if (POLL) clearInterval(POLL);
  POLL = setInterval(async () => {
    const j = await api(`/api/job?id=${r.id}`);
    if (j.error) return;
    $("#log").textContent = j.log || "(no output yet)";
    $("#log").scrollTop = $("#log").scrollHeight;
    const state = $("#run-state");
    state.textContent = j.running ? `${j.cmd} · ${j.elapsed}s`
      : `${j.cmd} finished (exit ${j.exit_code}) in ${j.elapsed}s`;
    state.classList.toggle("live", !!j.running);
    if (!j.running) { clearInterval(POLL); POLL = null; LOADERS.overview(); }
  }, 1500);
});

/* ─────────────────────────────── boot ───────────────────────────────── */
const VIEWS = new Set($$(".view").map(v => v.dataset.view));
const fromHash = () => {
  const v = (location.hash || "").slice(1);
  return VIEWS.has(v) ? v : "overview";
};
addEventListener("hashchange", () => go(fromHash()));

/* The provider badge belongs to the shell, not to a view, so it loads once on
   boot rather than only when the overview happens to be open. */
(async () => {
  const st = await api("/api/status");
  if (st.error) { $("#provider-text").textContent = "server unreachable"; return; }
  const p = st.provider || {};
  $("#provider-text").textContent = `${p.provider ?? "?"} · ${(p.model ?? "").split("/").pop()}`;
  $("#provider").classList.add("ok");
})();

go(fromHash());
setInterval(() => { if (document.documentElement.dataset.view === "overview") LOADERS.overview(); },
  20000);
