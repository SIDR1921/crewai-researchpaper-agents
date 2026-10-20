"""Stage 6: distil the API Extractor into a small local model with QLoRA.

The training set is built by running the API Extractor over the *training-split*
papers and keeping only the extractions where **every** claim passed the
Verifier. That is the point of the stage: the Verifier is the data filter, so the
student never learns to hallucinate evidence the teacher got away with.

4-bit QLoRA needs ``bitsandbytes``, which is CUDA-only, so the fine-tune itself is
a Colab-T4 job (notebook 05). The data building, prompt format and the evaluation
comparison all run anywhere, including on this Mac, and
:func:`train_qlora` degrades to plain LoRA in fp16/bf16 when 4-bit is unavailable
rather than pretending it quantised.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from .config import HF_MODELS
from .extract import SYSTEM, build_prompt


def build_dataset(records: dict[str, dict], verdicts: dict[str, list[dict]],
                  out_path: str | Path, require_all_supported: bool = True,
                  allow_partial: bool = False, seed: int = 0) -> dict:
    """Verifier-filtered (prompt, completion) pairs in chat format.

    ``records``: :func:`litreview.extract.load_records` output.
    ``verdicts``: ``paper_id -> [ClaimVerdict dicts]`` from the Verifier.
    """
    ok_kinds = {"supported", "partially"} if allow_partial else {"supported"}
    rows, stats = [], {"papers": len(records), "kept": 0, "dropped_unverified": 0,
                       "dropped_no_claims": 0, "dropped_no_input": 0}
    for pid, row in records.items():
        rec = row.get("record") or {}
        src = (row.get("meta") or {}).get("input_text") or ""
        vs = verdicts.get(pid) or []
        if not rec.get("claims"):
            stats["dropped_no_claims"] += 1
            continue
        if not src:
            stats["dropped_no_input"] += 1
            continue
        if require_all_supported and (not vs or not all(v["verdict"] in ok_kinds for v in vs)):
            stats["dropped_unverified"] += 1
            continue
        paper = {"paper_id": pid, "title": rec.get("title"), "year": rec.get("year"),
                 "sections": {}, "abstract": ""}
        target = json.dumps({k: v for k, v in rec.items()
                             if k in ("paper_id", "title", "year", "task", "methods",
                                      "datasets", "metrics", "claims", "limitations")},
                            ensure_ascii=False)
        rows.append({"paper_id": pid,
                     "messages": [
                         {"role": "system", "content": SYSTEM},
                         {"role": "user", "content": build_prompt(paper, src)},
                         {"role": "assistant", "content": target}]})
        stats["kept"] += 1
    random.Random(seed).shuffle(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats |= {"path": str(out_path), "require_all_supported": require_all_supported,
              "allow_partial": allow_partial,
              "mean_completion_chars": round(
                  sum(len(r["messages"][2]["content"]) for r in rows) / max(1, len(rows)), 1)}
    return stats


def train_qlora(dataset_path: str | Path, out_dir: str | Path,
                base_model: str = HF_MODELS["distill_base"], epochs: int = 2,
                lr: float = 2e-4, batch_size: int = 1, grad_accum: int = 8,
                max_len: int = 3072, lora_r: int = 16, lora_alpha: int = 32,
                seed: int = 0) -> dict:
    """QLoRA fine-tune. Returns a report; raises only on a genuine failure."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    rows = [json.loads(l) for l in Path(dataset_path).open()]
    if not rows:
        raise ValueError(f"no training rows in {dataset_path}")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quantised = False
    kwargs: dict = {"dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            quantised = True
        except Exception:
            quantised = False
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if quantised:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def encode(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        enc = tok(text, truncation=True, max_length=max_len)
        enc["labels"] = list(enc["input_ids"])
        return enc

    ds = Dataset.from_list(rows).map(encode, remove_columns=["messages", "paper_id"])
    out_dir = Path(out_dir)
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=5, save_strategy="epoch", save_total_limit=1,
        fp16=torch.cuda.is_available(), bf16=False, report_to=[], seed=seed,
        gradient_checkpointing=True, optim="paged_adamw_8bit" if quantised else "adamw_torch")
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    hist = trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    report = {"base_model": base_model, "quantised_4bit": quantised,
              "n_rows": len(rows), "epochs": epochs, "lr": lr, "max_len": max_len,
              "lora_r": lora_r, "lora_alpha": lora_alpha,
              "trainable_params": trainable, "out_dir": str(out_dir),
              "train_loss": round(float(hist.training_loss), 4) if hist else None,
              "device": "cuda" if torch.cuda.is_available() else "cpu/mps"}
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    return report


def compare_extractors(papers: Sequence[dict], api_extractor, distilled_extractor,
                       verifier=None, out_path: str | Path | None = None) -> dict:
    """Head-to-head on held-out papers: JSON validity, faithfulness, field agreement, cost."""
    from rapidfuzz import fuzz

    def run(ex) -> dict:
        recs, meta = {}, []
        for p in papers:
            rec, m = ex.extract(p)
            pid = p.get("paper_id") or p.get("arxiv_id")
            recs[pid] = (rec, m)
            meta.append(m)
        valid = sum(1 for r, _ in recs.values() if r is not None)
        lat = [m.get("latency_s") or 0.0 for m in meta]
        out = {"backend": getattr(ex, "name", "?"), "n_papers": len(papers),
               "json_valid": valid,
               "json_validity_rate": round(valid / max(1, len(papers)), 4),
               "latency_s_mean": round(sum(lat) / max(1, len(lat)), 2),
               "claims_per_paper": round(
                   sum(len(r.claims) for r, _ in recs.values() if r) / max(1, valid), 2)}
        if verifier is not None:
            rows = {pid: {"record": r.model_dump(mode="json"),
                          "meta": {"input_text": m.get("input_text", "")}}
                    for pid, (r, m) in recs.items() if r}
            if rows:
                vr = verifier.verify_corpus(rows)
                out |= {"claim_verdict_rates": vr["verdict_rates"],
                        "quote_found_rate": vr["quote_found_rate"],
                        "mean_entail_prob": vr["mean_entail_prob"]}
        return out | {"_records": recs}

    a, b = run(api_extractor), run(distilled_extractor)

    # field-level agreement on the papers both produced a record for
    both = [pid for pid in a["_records"]
            if a["_records"][pid][0] is not None and b["_records"].get(pid, (None,))[0] is not None]
    agree = {"n_compared": len(both)}
    for field in ("task", "methods", "datasets"):
        sims = []
        for pid in both:
            ra = a["_records"][pid][0].model_dump()
            rb = b["_records"][pid][0].model_dump()
            va, vb = ra.get(field), rb.get(field)
            if isinstance(va, list):
                if not va and not vb:
                    continue
                sa = {str(x).lower() for x in va}
                sb = {str(x).lower() for x in (vb or [])}
                sims.append(len(sa & sb) / max(1, len(sa | sb)))
            else:
                sims.append(fuzz.token_set_ratio(str(va or ""), str(vb or "")) / 100)
        agree[field] = round(sum(sims) / len(sims), 4) if sims else None

    from .llm import usage_summary
    report = {"api": {k: v for k, v in a.items() if k != "_records"},
              "distilled": {k: v for k, v in b.items() if k != "_records"},
              "field_agreement": agree,
              "llm_usage": usage_summary(),
              "note": ("The distilled model runs locally, so its monetary cost is zero and its "
                       "latency is hardware-bound; the API column's cost comes from the central "
                       "usage log.")}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
    return report
