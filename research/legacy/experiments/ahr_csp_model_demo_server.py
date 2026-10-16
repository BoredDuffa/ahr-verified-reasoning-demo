#!/usr/bin/env python3
"""FastAPI demo for the real AHR-CSP operation-synthesis checkpoint.

This intentionally uses the PyTorch AHR-CSP checkpoint through
experiments/ahr_operation_synthesis_playground.py. It is separate from the
static GitHub Pages demo, because GitHub Pages cannot execute the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ahr_csp_nl_pipeline import QwenJsonTranslator, response_contract, validate_dsl  # type: ignore
from ahr_csp_demo_program_backend import MULTIFAMILY_RANKER_BACKEND_TYPE, MULTIFAMILY_RANKER_MODEL_PATH, RANKER_MODEL_PATH  # type: ignore
from ahr_csp_demo_program_backend import load_multifamily_ranker_model, load_ranker_model  # type: ignore
from ahr_csp_demo_program_backend import SUPPORTED_FAMILIES as DEMO_PROGRAM_FAMILIES  # type: ignore
from ahr_csp_demo_program_backend import solve as solve_demo_program  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
OP_SYNTH_DATA_DIR = ROOT / "data" / "reasoning_challenge_v0" / "compiled_ahr_csp_v0"
DEFAULT_CHECKPOINT = "phase5r16x_operation_synthesis_len3_unhinted65_train_stability_fullgpu_b64"
DEFAULT_DATASET = "reasoning_curriculum_v16x_operation_synthesis_len3_unhinted65_4096_1024_1024.jsonl"
AHR_CSP_PARAMETER_COUNT = 2_751_495
AUX_RANKER_PARAMETER_COUNT = 757

app = FastAPI(title="AHR-CSP Model Demo", version="0.1")
APP_STATE = {
    "checkpoint": DEFAULT_CHECKPOINT,
    "dataset": DEFAULT_DATASET,
    "python": sys.executable,
}
TRANSLATOR = QwenJsonTranslator()


def answer_free_text(question: str) -> dict[str, Any]:
    translation = TRANSLATOR.translate(question)
    valid, reason = validate_dsl(translation.dsl)
    base_trace = [
        f"Translator source: {translation.source}.",
        f"Translator model: {translation.translator_model}.",
        "Translator emitted DSL only; it did not emit an answer.",
        f"DSL validation result: {reason}.",
    ]
    if not question.strip():
        return response_contract(
            question,
            translation,
            checkpoint=str(APP_STATE["checkpoint"]),
            verified=False,
            answer=None,
            trace=base_trace + ["No question text received."],
            abstain_reason="empty_question",
        )
    if not translation.valid_json:
        return response_contract(
            question,
            translation,
            checkpoint=str(APP_STATE["checkpoint"]),
            verified=False,
            answer=None,
            trace=base_trace,
            abstain_reason="translator_invalid_json",
        )
    if not valid:
        return response_contract(
            question,
            translation,
            checkpoint=str(APP_STATE["checkpoint"]),
            verified=False,
            answer=None,
            trace=base_trace,
            abstain_reason=reason,
        )
    family = translation.dsl.get("family")
    if family == "unsupported":
        return response_contract(
            question,
            translation,
            checkpoint=str(APP_STATE["checkpoint"]),
            verified=False,
            answer=None,
            trace=base_trace + ["The question is outside the current AHR-CSP demo scope."],
            abstain_reason="unsupported_question",
        )
    if reason == "known_but_untrained":
        return response_contract(
            question,
            translation,
            checkpoint=str(APP_STATE["checkpoint"]),
            verified=False,
            answer=None,
            trace=base_trace + [
                f"Family {family} is recognized as a future AHR-CSP family.",
                "No trained, gated AHR-CSP checkpoint is published for this family yet.",
            ],
            abstain_reason="ahr_family_not_trained_or_gated",
        )
    if reason == "demo_program_backend" and family in DEMO_PROGRAM_FAMILIES:
        result = solve_demo_program(question, translation.dsl)
        uses_5m = result.get("backend_type") == MULTIFAMILY_RANKER_BACKEND_TYPE
        family_backend_note = (
            f"Using trained 5M AHR multi-family ranker {result.get('ranker_checkpoint')} to select the legal program."
            if uses_5m
            else f"This family is not yet trained into the 2.75M operation-synthesis checkpoint; using auxiliary verified program backend {result['checkpoint']}."
        )
        trace = base_trace + [
            f"Built finite AHR-CSP demo program task for family {family}.",
            f"Primary AHR-CSP checkpoint loaded on this demo: {APP_STATE['checkpoint']} ({AHR_CSP_PARAMETER_COUNT:,} parameters).",
            family_backend_note,
        ] + list(result.get("trace", []))
        response = response_contract(
            question,
            translation,
            checkpoint=str(APP_STATE["checkpoint"]),
            verified=bool(result.get("verified")),
            answer=result.get("answer"),
            trace=trace,
            abstain_reason=result.get("abstain_reason"),
            model_results=result.get("model_results", []),
            backend_type=str(result.get("backend_type", "verified_program_backend")),
        )
        response["ahr_checkpoint_parameter_count"] = AHR_CSP_PARAMETER_COUNT
        response["aux_backend_checkpoint"] = str(result["checkpoint"])
        response["aux_ranker_parameter_count"] = result.get("ranker_parameter_count") or (AUX_RANKER_PARAMETER_COUNT if result.get("ranker_checkpoint") else None)
        response["uses_2_75m_checkpoint_for_reasoning"] = False
        response["uses_5m_multifamily_ranker_for_reasoning"] = uses_5m
        response["multifamily_ranker_checkpoint"] = result.get("ranker_checkpoint") if uses_5m else None
        response["multifamily_ranker_parameter_count"] = result.get("ranker_parameter_count") if uses_5m else None
        return response

    dsl = translation.dsl
    result = summarize(
        run_playground(
            count=int(dsl.get("count", 1)),
            split=str(dsl.get("split", "test")),
            family=str(dsl.get("task_family", "operation_synthesis_len3_factored")),
            seed="typed-demo",
        )
    )
    verified = result["records"] > 0 and result["cpu_valid"] == result["records"] and result["zero_energy"] == result["records"]
    first = result["results"][0] if result["results"] else {}
    answer = None
    abstain_reason = None
    trace = base_trace + [
        "Built AHR-CSP operation-synthesis request from validated DSL.",
        f"Ran checkpoint {result['checkpoint']}.",
        f"CPU verifier accepted {result['cpu_valid']}/{result['records']} records.",
        f"Zero hard energy on {result['zero_energy']}/{result['records']} records.",
    ]
    if verified:
        answer = (
            f"Ran {result['records']} operation-synthesis record(s): "
            f"{result['cpu_valid']}/{result['records']} CPU-valid, "
            f"{result['zero_energy']}/{result['records']} zero-energy. "
            f"First model program: {' -> '.join(first.get('model_program', []))}."
        )
    else:
        abstain_reason = "ahr_verifier_rejected_output"
        trace.append("Verifier gate failed, so no final answer is returned.")
    return response_contract(
        question,
        translation,
        checkpoint=str(result["checkpoint"]),
        verified=verified,
        answer=answer,
        trace=trace,
        abstain_reason=abstain_reason,
        model_results=result["results"],
        backend_type="trained_ahr_checkpoint",
    ) | {
        "ahr_checkpoint_parameter_count": AHR_CSP_PARAMETER_COUNT,
        "aux_backend_checkpoint": None,
        "aux_ranker_parameter_count": None,
        "uses_2_75m_checkpoint_for_reasoning": True,
        "uses_5m_multifamily_ranker_for_reasoning": False,
        "multifamily_ranker_checkpoint": None,
        "multifamily_ranker_parameter_count": None,
    }


def _extract_json(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError(f"playground did not return JSON: {stdout[-500:]}")
    return json.loads(stdout[start:])


def run_playground(count: int, split: str, family: str, seed: str, problem_id: str = "") -> dict[str, Any]:
    count = max(1, min(int(count), 25))
    cmd = [
        APP_STATE["python"],
        str(ROOT / "experiments" / "ahr_operation_synthesis_playground.py"),
        "--checkpoint",
        str(APP_STATE["checkpoint"]),
        "--dataset",
        str(APP_STATE["dataset"]),
        "--split",
        split,
        "--family",
        family,
        "--count",
        str(count),
        "--json",
    ]
    if problem_id:
        cmd.extend(["--problem-id", problem_id])
    else:
        cmd.extend(["--random", "--seed", seed])
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=120, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:] or f"playground exited {proc.returncode}")
    return _extract_json(proc.stdout)


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("results", [])
    valid = sum(1 for row in rows if row.get("cpu_check", {}).get("valid") is True)
    zero_energy = sum(1 for row in rows if row.get("cpu_check", {}).get("energy") == 0.0)
    return {
        "clinical_use": "forbidden",
        "checkpoint": result.get("checkpoint"),
        "checkpoint_parameter_count": AHR_CSP_PARAMETER_COUNT,
        "dataset": result.get("dataset"),
        "records": len(rows),
        "cpu_valid": valid,
        "zero_energy": zero_energy,
        "results": rows,
    }


def resolve_operation_dataset(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = OP_SYNTH_DATA_DIR / value
    if candidate.exists():
        return candidate
    return ROOT / "data" / value


def health_payload() -> dict[str, Any]:
    ranker_model = load_ranker_model()
    multifamily_ranker_model = load_multifamily_ranker_model()
    checkpoint_path = ROOT / "checkpoints" / f"{APP_STATE['checkpoint']}_best.pt"
    dataset_path = resolve_operation_dataset(str(APP_STATE["dataset"]))
    ranker_acceptance = None if ranker_model is None else ranker_model.get("acceptance", {})
    multifamily_ranker_acceptance = None if multifamily_ranker_model is None else multifamily_ranker_model.get("acceptance", {})
    return {
        "clinical_use": "forbidden",
        "status": "ok",
        "translator": TRANSLATOR.status,
        "ranker_checkpoint": str(RANKER_MODEL_PATH),
        "ranker_loaded": ranker_model is not None,
        "ranker_acceptance": ranker_acceptance,
        "multifamily_ranker_checkpoint": str(MULTIFAMILY_RANKER_MODEL_PATH),
        "multifamily_ranker_loaded": multifamily_ranker_model is not None,
        "multifamily_ranker_acceptance": multifamily_ranker_acceptance,
        "multifamily_ranker_parameter_count": None if multifamily_ranker_model is None else multifamily_ranker_model.get("parameter_count"),
        "operation_synthesis_checkpoint": str(checkpoint_path),
        "operation_synthesis_checkpoint_name": str(APP_STATE["checkpoint"]),
        "operation_synthesis_checkpoint_parameter_count": AHR_CSP_PARAMETER_COUNT,
        "operation_synthesis_checkpoint_present": checkpoint_path.exists(),
        "operation_synthesis_dataset": str(dataset_path),
        "operation_synthesis_dataset_present": dataset_path.exists(),
        "aux_demo_ranker_parameter_count": AUX_RANKER_PARAMETER_COUNT,
        "answer_policy": "Qwen emits DSL only. The 2,751,495-parameter AHR-CSP checkpoint handles operation-synthesis tasks. The 5M AHR multi-family ranker selects legal typed programs for gated math/stats/logic families when its published checkpoint is present. Exact verifiers release answers only with hard_energy=0.",
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AHR-CSP Model Demo</title>
  <style>
    :root{color-scheme:light;--bg:#eef2f6;--ink:#172026;--muted:#60707d;--line:#d7e0e7;--accent:#096b72;--ok:#1d7f4f;--warn:#8a5a00;--warnbg:#fff6dc}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif}
    main{max-width:1180px;margin:0 auto;padding:24px;display:grid;gap:16px}
    header,.panel{background:white;border:1px solid var(--line);padding:18px;box-shadow:0 14px 38px rgba(20,35,45,.08)}
    h1{margin:0 0 6px;font-size:2rem} p{margin:0;color:var(--muted);line-height:1.45}
    h2{margin:0 0 12px;font-size:1.25rem} h3{margin:0;color:var(--ink)}
    .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:end} label{display:grid;gap:5px;font-weight:700;color:#33434d}
    input,select,button,textarea{font:inherit;min-height:40px;border:1px solid var(--line);padding:0 10px;background:#fbfdfe}
    textarea{width:100%;min-height:96px;padding:12px;resize:vertical}
    button{cursor:pointer;background:var(--accent);color:white;font-weight:800;border-color:var(--accent)}
    .status{font-weight:800;color:var(--ok)} pre{white-space:pre-wrap;overflow:auto;background:#fbfdfe;border:1px solid var(--line);padding:12px}
    .result{display:grid;gap:10px;border-top:1px solid var(--line);padding-top:14px;margin-top:14px}
    .pill{display:inline-flex;align-items:center;min-height:24px;padding:0 8px;border:1px solid #b7dcbf;background:#eff9f1;color:#14633b;font-weight:800}
    .pill.warn{border-color:#e4c15a;background:var(--warnbg);color:var(--warn)}
    .status-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px}
    .metric{border:1px solid var(--line);padding:10px;background:#fbfdfe}.metric strong{display:block;color:var(--ink);font-size:1.05rem}.metric span{color:var(--muted);font-size:.9rem}
    .sample-group{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}.sample-group h3{font-size:1rem;margin-bottom:8px}
    .samples{display:flex;gap:8px;flex-wrap:wrap}.samples button{background:#f5f8fb;color:#172026;border-color:var(--line);font-weight:700}
    .samples button.primary{background:#e7f4f5;border-color:#8fc7cc;color:#064f55}
    .samples button.abstain{background:#fff6dc;border-color:#e4c15a;color:#6f4a00}
    .answer-mode{border:1px solid var(--line);padding:10px;background:#fbfdfe}
    @media(max-width:760px){main{padding:12px}.status-grid{grid-template-columns:1fr 1fr}h1{font-size:1.6rem}}
  </style>
</head>
<body>
<main>
  <header>
    <h1>AHR-CSP checkpoint and 5M ranker demo</h1>
    <p>Primary operation model: phase5r16x_operation_synthesis_len3_unhinted65_train_stability_fullgpu_b64, 2,751,495 parameters. Multi-family model: a gated 5M AHR ranker selects legal typed programs for math/stats/logic. Qwen only translates questions into typed DSL; it never answers.</p>
    <div class="status-grid">
      <div class="metric"><strong>2,751,495</strong><span>AHR-CSP checkpoint params</span></div>
      <div class="metric"><strong>4,998,161</strong><span>multi-family AHR ranker params</span></div>
      <div class="metric"><strong>0 hard energy</strong><span>required before answer release</span></div>
      <div class="metric"><strong>Qwen DSL only</strong><span>translator cannot answer</span></div>
    </div>
  </header>
  <section class="panel">
    <h2>Ask the live demo</h2>
    <textarea id="question" placeholder="Example: Run the AHR-CSP model on 5 operation synthesis examples"></textarea>
    <div class="controls">
      <button id="ask">Run actual AHR-CSP model</button>
    </div>
    <div class="sample-group">
      <h3>Actual 2.75M checkpoint</h3>
      <div class="samples">
        <button class="primary" data-q="Run the AHR-CSP model on 3 operation synthesis examples">len3 sample</button>
        <button class="primary" data-q="Please use AHR CSP to show five length 3 operation synthesis test records">worded len3</button>
        <button class="primary" data-q="Run the AHR-CSP model on 10 operation synthesis examples">10-record test</button>
        <button class="primary" data-q="Run the AHR-CSP model on 5 len2 operation synthesis examples">len2 sample</button>
      </div>
    </div>
    <div class="sample-group">
      <h3>5M AHR multi-family ranker plus exact verifier</h3>
      <div class="samples">
        <button data-q="What is the probability of rolling a sum of seven with two dice?">dice probability</button>
        <button data-q="A fair coin is tossed 3 times. What is the probability of exactly 2 heads?">coin exactly</button>
        <button data-q="If I flip 50 coins, what is the chance there is a sequence of 5 heads in a row?">coin run</button>
        <button data-q="A bag has 5 red, 3 blue, and 2 green balls. Two are drawn without replacement. What is the probability both are red?">urn</button>
        <button data-q="A card is drawn from a standard deck. What is the probability of drawing a king or a heart?">cards</button>
        <button data-q="What is the mean and sample variance of 4, 8, 10, and 12?">sample variance</button>
        <button data-q="Alice is taller than Bob. Bob is taller than Cara. Who is tallest?">logic ordering</button>
      </div>
    </div>
    <div class="sample-group">
      <h3>Honest abstention checks</h3>
      <div class="samples">
        <button class="abstain" data-q="Solve this Sudoku puzzle.">Sudoku</button>
        <button class="abstain" data-q="Prove that there are infinitely many primes.">proof</button>
        <button class="abstain" data-q="What is the weather tomorrow?">weather</button>
      </div>
    </div>
    <div id="answer"></div>
  </section>
  <section class="panel">
    <h2>Actual AHR-CSP checkpoint sampler</h2>
    <div class="controls">
      <label>Split<select id="split"><option>test</option><option>val</option><option>train</option></select></label>
      <label>Family<select id="family"><option>operation_synthesis_len3_factored</option><option>operation_synthesis_len2_factored</option></select></label>
      <label>Count<input id="count" type="number" min="1" max="25" value="3"></label>
      <label>Seed<input id="seed" value="live-demo"></label>
      <button id="run">Run actual AHR-CSP model</button>
    </div>
    <p id="status"></p>
    <div id="out"></div>
  </section>
</main>
<script>
const out=document.getElementById('out'); const status=document.getElementById('status'); const answer=document.getElementById('answer'); const question=document.getElementById('question');
function esc(x){return String(x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function traceOf(v){
  if(!v) return [];
  if(Array.isArray(v.trace)) return v.trace;
  if(Array.isArray(v.proof)) return v.proof;
  return Object.entries(v).filter(([k])=>k!=='results').map(([k,val])=>`${k}: ${JSON.stringify(val)}`);
}
function renderAnswer(data){
  const trace=data.trace || traceOf(data.verification);
  const answerText = data.answer || `Abstained: ${data.abstain_reason || 'not verified'}`;
  const released = data.verified && data.answer ? 'yes' : 'no';
  const modeText = data.uses_2_75m_checkpoint_for_reasoning ? '2.75M AHR-CSP checkpoint did the neural reasoning' : (data.uses_5m_multifamily_ranker_for_reasoning ? '5M AHR multi-family ranker selected the verified program' : (data.aux_backend_checkpoint ? 'Auxiliary verified QA backend; not yet trained into the current AHR checkpoints' : 'No answer released'));
  const modeClass = (data.uses_2_75m_checkpoint_for_reasoning || data.uses_5m_multifamily_ranker_for_reasoning) ? '' : ' warn';
  answer.innerHTML = `<div class="result"><p><span class="pill">verified=${esc(data.verified)}</span> <span class="pill">answer_released=${esc(released)}</span></p>`+
    `<div class="answer-mode"><span class="pill${modeClass}">${esc(modeText)}</span></div>`+
    `<p>Translator: ${esc(data.translator_model || 'unknown')} (${esc(data.translator_source || 'unknown')}); DSL only, no answer generated by translator.</p>`+
    `<p><span class="pill">primary AHR-CSP checkpoint=${esc(data.ahr_checkpoint || 'unknown')}</span> params=${esc(data.ahr_checkpoint_parameter_count || 'unknown')}</p>`+
    `<p><span class="pill">backend=${esc(data.backend_type || 'unknown')}</span> uses_2.75M=${esc(data.uses_2_75m_checkpoint_for_reasoning)} uses_5M=${esc(data.uses_5m_multifamily_ranker_for_reasoning)}</p>`+
    (data.multifamily_ranker_checkpoint ? `<p>5M AHR ranker: ${esc(data.multifamily_ranker_checkpoint)}; params=${esc(data.multifamily_ranker_parameter_count || 'unknown')}</p>` : '')+
    (data.aux_backend_checkpoint && !data.uses_5m_multifamily_ranker_for_reasoning ? `<p>Auxiliary backend for this non-operation family: ${esc(data.aux_backend_checkpoint)}${data.aux_ranker_parameter_count ? `; auxiliary ranker params=${esc(data.aux_ranker_parameter_count)}` : ''}</p>` : '')+
    `<h3>${esc(answerText)}</h3>`+
    `<pre>Parsed DSL\\n${esc(JSON.stringify(data.dsl,null,2))}</pre>`+
    `<pre>${esc(trace.map((x,i)=>`${i+1}. ${x}`).join('\\n') || JSON.stringify(data,null,2))}</pre>`+
    (data.model_results && data.model_results.length ? `<pre>${esc(JSON.stringify(data.model_results.slice(0,3),null,2))}</pre>` : '')+
    `</div>`;
}
function render(data){
  out.innerHTML = `<p><span class="pill">clinical_use=${esc(data.clinical_use)}</span> checkpoint: ${esc(data.checkpoint)} (${esc(data.checkpoint_parameter_count || 'unknown')} params)</p>`+
    `<p class="status">${data.cpu_valid}/${data.records} CPU-valid, ${data.zero_energy}/${data.records} zero-energy</p>`;
  data.results.forEach((r,i)=>{
    out.innerHTML += `<div class="result"><strong>${i+1}. ${esc(r.id)}</strong>`+
      `<p>start=${esc(JSON.stringify(r.start_sequence))} target=${esc(JSON.stringify(r.target_sequence))}</p>`+
      `<p>model program: <strong>${esc((r.model_program||[]).join(' -> '))}</strong></p>`+
      `<p>verified: ${esc(r.cpu_check.valid)} energy=${esc(r.cpu_check.energy)}</p>`+
      `<pre>${esc(JSON.stringify({model_assignment:r.model_assignment, reference_assignment:r.reference_assignment, repair:r.repair},null,2))}</pre></div>`;
  });
}
document.querySelectorAll('.samples button').forEach(btn=>btn.onclick=()=>{question.value=btn.dataset.q; question.focus();});
document.getElementById('ask').onclick=async()=>{
  answer.innerHTML='<p>Running verified backend...</p>';
  const res=await fetch('/api/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:question.value})});
  const data=await res.json();
  if(!res.ok){answer.innerHTML='<pre>'+esc(JSON.stringify(data,null,2))+'</pre>'; return;}
  renderAnswer(data);
};
document.getElementById('run').onclick=async()=>{
  status.textContent='Running checkpoint. This can take a few seconds because the model is real PyTorch, not browser JS.';
  out.innerHTML='';
  const params=new URLSearchParams({split:split.value,family:family.value,count:count.value,seed:seed.value});
  const res=await fetch('/api/run?'+params.toString());
  const data=await res.json();
  if(!res.ok){status.textContent='Error'; out.innerHTML='<pre>'+esc(JSON.stringify(data,null,2))+'</pre>'; return;}
  status.textContent='Done';
  render(data);
};
</script>
</body>
</html>"""


@app.get("/api/run")
def api_run(count: int = 3, split: str = "test", family: str = "operation_synthesis_len3_factored", seed: str = "live-demo") -> JSONResponse:
    if split not in {"train", "val", "test"}:
        raise HTTPException(status_code=400, detail="split must be train, val, or test")
    if family not in {"operation_synthesis_len2_factored", "operation_synthesis_len3_factored"}:
        raise HTTPException(status_code=400, detail="unsupported family")
    try:
        return JSONResponse(summarize(run_playground(count, split, family, seed)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/health")
def api_health() -> JSONResponse:
    return JSONResponse(health_payload())


@app.post("/api/ask")
def api_ask(payload: dict[str, Any]) -> JSONResponse:
    question = str(payload.get("question", ""))
    try:
        answer = answer_free_text(question)
        answer["clinical_use"] = "forbidden"
        return JSONResponse(answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    APP_STATE["checkpoint"] = args.checkpoint
    APP_STATE["dataset"] = args.dataset
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
