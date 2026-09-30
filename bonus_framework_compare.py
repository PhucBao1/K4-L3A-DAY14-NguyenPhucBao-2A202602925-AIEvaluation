"""Exercise 3.4 (bonus) — compare RAGAS and DeepEval on the same 20 answers.

Both frameworks score the exact artifacts produced by the lab pipeline
(``golden_dataset.json`` + ``artifacts/actual_answers.json``) with the same
judge model, so differences come from the frameworks, not the inputs.

Run (needs OPENAI_API_KEY in .env and ``requirements-bonus.txt`` installed):
    .venv-bonus/Scripts/python.exe bonus_framework_compare.py
Output: artifacts/framework_comparison.json + Markdown tables on stdout.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).parent
JUDGE_MODEL = "gpt-4o-mini"
PASS_THRESHOLD = 0.5
# Common metric names used in the comparison, in both frameworks' terms.
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def load_cases() -> list[dict[str, Any]]:
    golden = json.loads((ROOT / "golden_dataset.json").read_text(encoding="utf-8"))
    actual = json.loads(
        (ROOT / "artifacts" / "actual_answers.json").read_text(encoding="utf-8")
    )
    answers = {record["id"]: record for record in actual["answers"]}
    return [
        {
            "id": pair["id"],
            "difficulty": pair["difficulty"],
            "question": pair["question"],
            "expected": pair["expected_answer"],
            "answer": answers[pair["id"]]["actual_answer"],
            "contexts": [c["text"] for c in answers[pair["id"]]["retrieved_contexts"]],
        }
        for pair in golden["qa_pairs"]
    ]


def load_heuristic_scores() -> dict[str, dict[str, float]]:
    path = ROOT / "artifacts" / "benchmark_results.json"
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    return {
        r["id"]: {
            "faithfulness": r["faithfulness"],
            "answer_relevancy": r["relevance"],
            "context_recall": r["context_recall"],
            "context_precision": r["context_precision"],
        }
        for r in results
    }


def run_ragas(cases: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": case["question"],
                "response": case["answer"],
                "retrieved_contexts": case["contexts"],
                "reference": case["expected"],
            }
            for case in cases
        ]
    )
    metric_objects = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]
    result = evaluate(
        dataset,
        metrics=metric_objects,
        llm=LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings()),
        show_progress=True,
    )
    frame = result.to_pandas()

    scores: dict[str, dict[str, float | None]] = {}
    for case, (_, row) in zip(cases, frame.iterrows()):
        scores[case["id"]] = {
            common: _as_score(row.get(metric.name))
            for common, metric in zip(METRICS, metric_objects)
        }
    return scores


def run_deepeval(
    cases: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, str]]]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    metric_classes = (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    scores: dict[str, dict[str, float | None]] = {}
    reasons: dict[str, dict[str, str]] = {}
    for index, case in enumerate(cases, start=1):
        test_case = LLMTestCase(
            input=case["question"],
            actual_output=case["answer"],
            expected_output=case["expected"],
            retrieval_context=case["contexts"],
        )
        scores[case["id"]], reasons[case["id"]] = {}, {}
        for common, metric_class in zip(METRICS, metric_classes):
            metric = metric_class(
                threshold=PASS_THRESHOLD,
                model=JUDGE_MODEL,
                include_reason=True,
                async_mode=False,
            )
            try:
                metric.measure(test_case)
                scores[case["id"]][common] = _as_score(metric.score)
                reasons[case["id"]][common] = str(metric.reason or "")
            except Exception as exc:  # one bad judge call must not stop the run
                scores[case["id"]][common] = None
                reasons[case["id"]][common] = f"ERROR: {exc}"
        print(f"[DeepEval] {index:02d}/{len(cases)} {case['id']} done", flush=True)
    return scores, reasons


def _as_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return None if score != score else score  # NaN → None


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank), for Spearman correlation."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rx, ry = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def failing_ids(scores: dict[str, dict[str, float | None]]) -> list[str]:
    """A case fails when faithfulness or answer relevancy < PASS_THRESHOLD."""
    return [
        case_id
        for case_id, row in scores.items()
        if any(
            row.get(m) is not None and row[m] < PASS_THRESHOLD
            for m in ("faithfulness", "answer_relevancy")
        )
    ]


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing — add it to .env")

    cases = load_cases()
    heuristic = load_heuristic_scores()

    started = time.perf_counter()
    ragas_scores = run_ragas(cases)
    ragas_seconds = time.perf_counter() - started

    started = time.perf_counter()
    deepeval_scores, deepeval_reasons = run_deepeval(cases)
    deepeval_seconds = time.perf_counter() - started

    frameworks = {
        "heuristic": heuristic,
        "ragas": ragas_scores,
        "deepeval": deepeval_scores,
    }
    ids = [case["id"] for case in cases]
    averages = {
        name: {m: _mean([scores[i].get(m) for i in ids]) for m in METRICS}
        for name, scores in frameworks.items()
    }
    correlations = {
        m: {
            "ragas_vs_deepeval": spearman(
                [ragas_scores[i].get(m) for i in ids],
                [deepeval_scores[i].get(m) for i in ids],
            ),
            "heuristic_vs_ragas": spearman(
                [heuristic[i].get(m) for i in ids],
                [ragas_scores[i].get(m) for i in ids],
            ),
            "heuristic_vs_deepeval": spearman(
                [heuristic[i].get(m) for i in ids],
                [deepeval_scores[i].get(m) for i in ids],
            ),
        }
        for m in METRICS
    }
    failures = {name: failing_ids(scores) for name, scores in frameworks.items()}

    output = {
        "judge_model": JUDGE_MODEL,
        "pass_threshold": PASS_THRESHOLD,
        "runtime_seconds": {
            "ragas": round(ragas_seconds, 1),
            "deepeval": round(deepeval_seconds, 1),
        },
        "averages": averages,
        "spearman": correlations,
        "failing_ids": failures,
        "per_case": {
            i: {name: frameworks[name][i] for name in frameworks} for i in ids
        },
        "deepeval_reasons": deepeval_reasons,
    }
    out_path = ROOT / "artifacts" / "framework_comparison.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n| Metric | Heuristic | RAGAS | DeepEval | Spearman RAGAS~DeepEval |")
    print("|---|---:|---:|---:|---:|")
    for m in METRICS:
        print(
            f"| {m} | {_fmt(averages['heuristic'][m])} | {_fmt(averages['ragas'][m])} | "
            f"{_fmt(averages['deepeval'][m])} | {_fmt(correlations[m]['ragas_vs_deepeval'])} |"
        )
    print("\n| ID | RAGAS F | RAGAS AR | DeepEval F | DeepEval AR | Heuristic F | Heuristic AR |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for i in ids:
        r, d, h = ragas_scores[i], deepeval_scores[i], heuristic[i]
        print(
            f"| {i} | {_fmt(r['faithfulness'])} | {_fmt(r['answer_relevancy'])} | "
            f"{_fmt(d['faithfulness'])} | {_fmt(d['answer_relevancy'])} | "
            f"{_fmt(h['faithfulness'])} | {_fmt(h['answer_relevancy'])} |"
        )
    print(f"\nFailing (F or AR < {PASS_THRESHOLD}):")
    for name, fail_ids in failures.items():
        print(f"  {name}: {len(fail_ids)} -> {fail_ids}")
    print(f"\nRuntime: RAGAS {ragas_seconds:.1f}s | DeepEval {deepeval_seconds:.1f}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
