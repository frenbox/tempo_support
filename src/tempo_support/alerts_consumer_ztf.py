import io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import fastavro
import pandas as pd
import requests
from confluent_kafka import Consumer
from dotenv import load_dotenv
from pymongo import MongoClient

from tempo_support.slack_post import format_message, post_to_slack
from tempo_support.tempo_boom_ztf import get_taxonomy, leaf_class_probs, run_tempo

LOG_FILE = "tempo_ztf.log"
KAFKA_TOPIC = "ZTF_alerts_results"
FILTER_NAME = "superphot_ztf"
MODEL_TITLE = "TEMPO"
FRITZ_BASE_URL = "https://fritz.science"
RESULTS_CSV = Path("results") / "tempo_ztf_results.csv"
POST_TO_SLACK = False  # Slack uploads paused; flip to True to re-enable.
FRITZ_ANNOTATION_ORIGIN = "tempo"
FRITZ_GROUP_IDS = [1973]  # UMN TEMPO — annotation visibility scope.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
for _noisy in ("matplotlib", "PIL", "kaleido", "choreographer"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv(Path.home() / ".env")
_mongo_user = os.getenv("BOOM_DATABASE__USERNAME")
_mongo_pass = os.getenv("BOOM_DATABASE__PASSWORD")
_mongo_url = (
    f"mongodb://{_mongo_user}:{_mongo_pass}@localhost:27017"
    if _mongo_user and _mongo_pass
    else "mongodb://localhost:27017"
)
_db = MongoClient(_mongo_url)["boom"]
alerts_aux_coll = _db["ZTF_alerts_aux"]


def _fritz_classifications(ztf_id):
    fritz_token = os.getenv("FRITZ_TOKEN")
    if not fritz_token:
        return []
    try:
        r = requests.get(
            f"{FRITZ_BASE_URL}/api/sources/{ztf_id}/classifications",
            headers={"Authorization": f"token {fritz_token}"},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        j = r.json()
        if j.get("status") != "success":
            return []
        return j.get("data") or []
    except Exception:
        logger.exception("[%s] Fritz classifications fetch failed", ztf_id)
        return []


def _format_fritz_block(classifications):
    if not classifications:
        return None
    parts = []
    for c in classifications[:5]:
        name = c.get("classification") or "?"
        prob = c.get("probability")
        if isinstance(prob, (int, float)):
            parts.append(f"{name} ({prob:.2f})")
        else:
            parts.append(name)
    return "*Fritz:* " + ", ".join(parts)


def _append_csv(ztf_id, result, fritz_classifications):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fritz_str = "; ".join(
        f"{c.get('classification', '?')}({c.get('probability')})"
        for c in (fritz_classifications or [])
    )
    leaf = result["leaf"]
    decision = result.get("decision", {})
    fallback = decision.get("fallback") or {}
    row = {
        "objectId": ztf_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_events": int(result.get("n_events", 0)),
        "horizon_days": float(result.get("final_horizon_days", float("nan"))),
        "leaf_pred": leaf.get("pred_thresholded_name"),
        "leaf_pred_prob": float(leaf.get("pred_thresholded_prob", float("nan"))),
        "leaf_kept": bool(leaf.get("kept")),
        "decision_label": fallback.get("label"),
        "decision_level": fallback.get("level_name"),
        "abstain_completely": bool(decision.get("abstain_completely")),
        "uncertainty_gate_name": leaf.get("uncertainty_selected_name"),
        "uncertainty_gate_value": float(leaf.get("uncertainty_selected_value", float("nan"))),
        "fritz_classifications": fritz_str,
    }
    for cname, p in zip(leaf["class_names"], leaf["probs_final"]):
        row[f"final_prob_{cname}"] = float(p)
    for cname, s in zip(leaf["class_names"], leaf.get("prob_std_calibrated") or []):
        row[f"calibrated_std_{cname}"] = float(s)
    for k, v in (leaf.get("score_map") or {}).items():
        row[f"score_{k}"] = float(v)
    if leaf.get("ood"):
        row["ood_score"] = float(leaf["ood"]["score"])
        row["ood_votes"] = int(leaf["ood"]["votes"])
        row["ood_flag"] = bool(leaf["ood"]["flag"])
    pd.DataFrame([row]).to_csv(
        RESULTS_CSV,
        mode="a",
        index=False,
        header=not RESULTS_CSV.exists(),
    )


def _fritz_url(ztf_id):
    fritz_token = os.getenv("FRITZ_TOKEN")
    if not fritz_token:
        return f"{FRITZ_BASE_URL}/alerts/ztf/{ztf_id}"
    try:
        r = requests.get(
            f"{FRITZ_BASE_URL}/api/sources/{ztf_id}",
            headers={"Authorization": f"token {fritz_token}"},
            timeout=10,
        )
        if r.status_code == 200 and r.json().get("status") == "success":
            return f"{FRITZ_BASE_URL}/source/{ztf_id}"
    except Exception:
        logger.debug("[%s] Fritz source check failed, falling back to alerts URL", ztf_id)
    return f"{FRITZ_BASE_URL}/alerts/ztf/{ztf_id}"


def _ordered_class_probs(class_probs, taxonomy):
    """Order leaf-class probabilities as a pre-order walk of the taxonomy.

    Classes are grouped by their hierarchy path (domain -> family -> class), so
    Transient/SN classes come before Variable classes. Keys not present in the
    taxonomy are appended at the end in their original order (defensive).
    """
    paths = getattr(taxonomy, "broad2hierarchy_path", {}) or {}
    known = sorted(
        (c for c in class_probs if c in paths),
        key=lambda c: tuple(paths.get(c, [c])),
    )
    ordered = {c: class_probs[c] for c in known}
    for k, v in class_probs.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _get_annotation_id(ztf_id, origin, headers):
    """Fetch the annotation ID for a source by origin, or None."""
    endpoint = f"{FRITZ_BASE_URL}/api/sources/{ztf_id}/annotations"
    try:
        resp_json = requests.get(endpoint, headers=headers, timeout=10).json()
        if resp_json.get("status") == "success":
            for ann in resp_json.get("data", []):
                if ann.get("origin") == origin:
                    return ann.get("annotation_id") or ann.get("id")
    except Exception:
        logger.exception("[%s] Failed to fetch existing annotations", ztf_id)
    return None


def annotate_fritz(class_probs, ztf_id, taxonomy, origin=FRITZ_ANNOTATION_ORIGIN,
                   group_ids=None, previous_annotation_id=None):
    """Post (or update) per-class probability annotations on Fritz.

    The probability dict is ordered as a pre-order taxonomy walk (Transient/SN
    classes, then Variable classes). On a duplicate-origin POST failure, falls
    back to fetching the existing annotation and updating it via PUT.

    Returns the annotation_id of the created/updated annotation, or None.
    """
    fritz_token = os.getenv("FRITZ_TOKEN")
    if not fritz_token:
        logger.warning("[%s] FRITZ_TOKEN not set, skipping annotation", ztf_id)
        return None

    headers = {
        "Authorization": f"token {fritz_token}",
        "Content-Type": "application/json",
    }

    data = {
        k: float(v) if isinstance(v, (int, float)) else v
        for k, v in _ordered_class_probs(class_probs, taxonomy).items()
    }
    payload = {"origin": origin, "data": data}
    if group_ids is not None:
        payload["group_ids"] = group_ids

    base = f"{FRITZ_BASE_URL}/api/sources/{ztf_id}/annotations"
    if previous_annotation_id is not None:
        response = requests.put(f"{base}/{previous_annotation_id}", json=payload,
                                headers=headers, timeout=10)
    else:
        response = requests.post(base, json=payload, headers=headers, timeout=10)

    resp_json = response.json()
    if resp_json.get("status") == "success":
        data_resp = resp_json.get("data", {})
        logger.info("[%s] Annotation saved.", ztf_id)
        return data_resp.get("annotation_id") or data_resp.get("id")

    # POST likely failed due to duplicate origin — fetch existing and PUT.
    logger.warning("[%s] Annotation POST failed: %s", ztf_id, resp_json.get("message"))
    if previous_annotation_id is None:
        existing_id = _get_annotation_id(ztf_id, origin, headers)
        if existing_id is not None:
            retry = requests.put(f"{base}/{existing_id}", json=payload,
                                 headers=headers, timeout=10).json()
            if retry.get("status") == "success":
                logger.info("[%s] Annotation updated via fallback PUT.", ztf_id)
                return existing_id
            logger.error("[%s] Fallback PUT failed: %s", ztf_id, retry.get("message"))
    return None


def read_avro(msg):
    bytes_io = io.BytesIO(msg.value())
    bytes_io.seek(0)
    for record in fastavro.reader(bytes_io):
        return record
    return None


consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "umn_boom_kafka_consumer_group_tempo_ztf",
    "auto.offset.reset": "latest",
    "enable.auto.commit": False,
    "session.timeout.ms": 6000,
    "max.poll.interval.ms": 300000,
    "security.protocol": "PLAINTEXT",
})
consumer.subscribe([KAFKA_TOPIC])
logger.info("Subscribed to topic: %s", KAFKA_TOPIC)


def consume():
    logger.info("Listening for messages...")
    total_consumed = 0
    consecutive_empty_polls = 0

    try:
        while True:
            msg = consumer.poll(timeout=10.0)
            if msg is None:
                consecutive_empty_polls += 1
                if consecutive_empty_polls % 6 == 1:
                    logger.info(
                        "No new messages (idle ~%ds, consumed %d)",
                        consecutive_empty_polls * 10,
                        total_consumed,
                    )
                continue
            consecutive_empty_polls = 0
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue

            record = read_avro(msg)
            if record is None:
                logger.error("Failed to deserialize Avro at offset %s", msg.offset())
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            ztf_id = record["objectId"]

            passes_filter = any(
                FILTER_NAME in f["filter_name"] for f in record.get("filters") or []
            )
            if not passes_filter:
                logger.debug("[%s] did not pass %s, skipping", ztf_id, FILTER_NAME)
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            aux_doc = alerts_aux_coll.find_one({"_id": ztf_id})
            if aux_doc is None:
                logger.warning("[%s] no aux doc in Mongo, skipping", ztf_id)
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            prv_candidates = aux_doc.get("prv_candidates") or []
            if not prv_candidates:
                logger.warning("[%s] no prv_candidates in aux doc, skipping", ztf_id)
                total_consumed += 1
                consumer.commit(message=msg)
                continue

            logger.info("[%s] Running TEMPO (prv=%d)", ztf_id, len(prv_candidates))

            try:
                result = run_tempo(ztf_id=ztf_id, prv_candidates=prv_candidates)
            except Exception:
                logger.exception("[%s] run_tempo failed", ztf_id)
                result = None

            if result is not None:
                link = _fritz_url(ztf_id)
                fritz_classifications = _fritz_classifications(ztf_id)
                _append_csv(ztf_id, result, fritz_classifications)

                try:
                    annotation_id = annotate_fritz(
                        leaf_class_probs(result), ztf_id, get_taxonomy(),
                        group_ids=FRITZ_GROUP_IDS,
                    )
                    logger.info("[%s] Fritz annotation: %s", ztf_id, annotation_id)
                except Exception:
                    logger.exception("[%s] Fritz annotation failed", ztf_id)

                fritz_block = _format_fritz_block(fritz_classifications)
                logger.info(
                    "[%s] classification:\n%s",
                    ztf_id,
                    format_message(
                        ztf_id, result, title=MODEL_TITLE, link=link, extra_text=fritz_block
                    ),
                )

                if not POST_TO_SLACK:
                    logger.debug("[%s] Slack posting disabled, skipping", ztf_id)
                elif not fritz_classifications:
                    logger.info("[%s] no Fritz classification, skipping Slack post", ztf_id)
                else:
                    try:
                        file_id = post_to_slack(
                            ztf_id,
                            result,
                            title=MODEL_TITLE,
                            link=link,
                            channel_env="SLACK_TEMPO_CHANNEL_ID",
                            extra_text=fritz_block,
                        )
                        if file_id:
                            logger.info("[%s] posted to Slack (file_id=%s)", ztf_id, file_id)
                    except Exception:
                        logger.exception("[%s] Slack post failed", ztf_id)
            else:
                logger.warning("[%s] no result", ztf_id)

            total_consumed += 1
            consumer.commit(message=msg)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        logger.info("Processed %d messages.", total_consumed)
        consumer.close()


if __name__ == "__main__":
    consume()
