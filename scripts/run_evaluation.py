"""MLflow Evaluation script for the CeramiCraft Customer Support Agent.

Usage:
    uv run python scripts/run_evaluation.py

Environment variables:
    MLFLOW_TRACKING_URI   MLflow server URL (required)
    AGENT_URL             Agent base URL (default: http://localhost:8080)
    AGENT_TOKEN           Bearer token for agent auth (default: test-token)
"""

import csv
import json
import logging
import os
import statistics
import time
import uuid
from typing import Any

import httpx
import mlflow

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8080")
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "test-token")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

# ---------------------------------------------------------------------------
# Test dataset — covers key scenarios across all subgraphs
# ---------------------------------------------------------------------------
EVAL_DATASET: list[dict[str, Any]] = [
    # --- browse ---
    {
        "input": "你们有陶瓷花瓶吗？",
        "expected_keywords": ["花瓶", "陶瓷"],
        "scenario": "browse",
    },
    {
        "input": "帮我找价格在100元以内的陶瓷产品",
        "expected_keywords": ["产品", "元"],
        "scenario": "browse",
    },
    {
        "input": "有没有适合送礼的陶瓷制品",
        "expected_keywords": ["礼", "陶瓷"],
        "scenario": "browse",
    },
    # --- order (auth required) ---
    {
        "input": "我想查看我的订单",
        "expected_keywords": ["订单", "登录"],
        "scenario": "order",
    },
    {
        "input": "我的订单什么时候发货",
        "expected_keywords": ["订单", "登录"],
        "scenario": "order",
    },
    # --- cart (auth required) ---
    {
        "input": "我想把一个陶瓷杯加入购物车",
        "expected_keywords": ["购物车", "登录"],
        "scenario": "cart",
    },
    # --- account (auth required) ---
    {
        "input": "我想修改我的收货地址",
        "expected_keywords": ["地址", "登录"],
        "scenario": "account",
    },
    # --- chitchat ---
    {"input": "你好", "expected_keywords": ["你好", "帮"], "scenario": "chitchat"},
    {
        "input": "你们是做什么的",
        "expected_keywords": ["陶瓷", "CeramiCraft"],
        "scenario": "chitchat",
    },
    # --- unsupported ---
    {
        "input": "我要付款",
        "expected_keywords": ["暂", "网站"],
        "scenario": "unsupported",
    },
    {
        "input": "帮我退款",
        "expected_keywords": ["暂", "网站"],
        "scenario": "unsupported",
    },
    # --- multilingual ---
    {
        "input": "Do you have ceramic plates?",
        "expected_keywords": ["ceramic", "plate"],
        "scenario": "browse_en",
    },
]


# ---------------------------------------------------------------------------
# Agent caller
# ---------------------------------------------------------------------------


def call_agent(message: str) -> tuple[str, float]:
    thread_id = str(uuid.uuid4())
    t0 = time.time()
    resp = httpx.post(
        f"{AGENT_URL}/chat",
        json={"message": message, "thread_id": thread_id},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AGENT_TOKEN}",
        },
        timeout=60.0,
    )
    latency = time.time() - t0
    resp.raise_for_status()
    return resp.json()["reply"], latency


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def keyword_hit_rate(reply: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    reply_lower = reply.lower()
    return sum(1 for kw in keywords if kw.lower() in reply_lower) / len(keywords)


def reply_length_score(reply: str) -> float:
    n = len(reply)
    if n < 20:
        return 0.0
    if n > 600:
        return max(0.0, 1.0 - (n - 600) / 600)
    return 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI is required")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ceramicraft-cs-agent-eval")

    logger.info(
        "Starting evaluation — %d cases against %s", len(EVAL_DATASET), AGENT_URL
    )

    rows: list[dict] = []
    for case in EVAL_DATASET:
        logger.info("  [%s] %s", case["scenario"], case["input"][:50])
        try:
            reply, latency = call_agent(case["input"])
            khr = keyword_hit_rate(reply, case["expected_keywords"])
            rls = reply_length_score(reply)
            rows.append(
                {
                    "scenario": case["scenario"],
                    "input": case["input"],
                    "reply": reply,
                    "expected_keywords": json.dumps(
                        case["expected_keywords"], ensure_ascii=False
                    ),
                    "keyword_hit_rate": khr,
                    "reply_length_score": rls,
                    "composite_score": (khr + rls) / 2,
                    "latency_s": round(latency, 3),
                    "error": "",
                }
            )
        except Exception as exc:
            logger.warning("  FAILED: %s", exc)
            rows.append(
                {
                    "scenario": case["scenario"],
                    "input": case["input"],
                    "reply": "",
                    "expected_keywords": json.dumps(
                        case["expected_keywords"], ensure_ascii=False
                    ),
                    "keyword_hit_rate": 0.0,
                    "reply_length_score": 0.0,
                    "composite_score": 0.0,
                    "latency_s": 0.0,
                    "error": str(exc),
                }
            )

    # Aggregate metrics
    def mean(key: str) -> float:
        vals = [r[key] for r in rows]
        return statistics.mean(vals) if vals else 0.0

    def quantile95(key: str) -> float:
        vals = sorted(r[key] for r in rows)
        if not vals:
            return 0.0
        idx = int(len(vals) * 0.95)
        return vals[min(idx, len(vals) - 1)]

    agg_metrics = {
        "keyword_hit_rate_mean": mean("keyword_hit_rate"),
        "reply_length_score_mean": mean("reply_length_score"),
        "composite_score_mean": mean("composite_score"),
        "latency_s_mean": mean("latency_s"),
        "latency_s_p95": quantile95("latency_s"),
        "error_count": float(sum(1 for r in rows if r["error"])),
        "total_cases": float(len(rows)),
    }

    with mlflow.start_run(run_name=f"eval-{time.strftime('%Y%m%d-%H%M')}"):
        mlflow.log_params(
            {
                "agent_url": AGENT_URL,
                "total_cases": len(EVAL_DATASET),
                "dataset_version": "v1",
            }
        )
        mlflow.log_metrics(agg_metrics)

        # Save per-case CSV as artifact
        results_path = "/tmp/eval_results.csv"
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        mlflow.log_artifact(results_path, artifact_path="eval")

    logger.info("=== Results ===")
    for k, v in agg_metrics.items():
        logger.info("  %s: %.3f", k, v)
    logger.info("Logged to MLflow experiment: ceramicraft-cs-agent-eval")


if __name__ == "__main__":
    main()
