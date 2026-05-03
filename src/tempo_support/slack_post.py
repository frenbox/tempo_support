"""Post TEMPO classification results to Slack."""
import logging
import math
import os
import tempfile
from pathlib import Path

import requests

from tempo_support.plot_tempo import save_sunburst
from tempo_support.tempo_boom_ztf import get_taxonomy

logger = logging.getLogger(__name__)

SLACK_GET_URL = "https://slack.com/api/files.getUploadURLExternal"
SLACK_COMPLETE_URL = "https://slack.com/api/files.completeUploadExternal"

LABEL_W = 12
VALUE_W = 9


def _fmt_pct(p):
    if p is None:
        return f"{'n/a':>{VALUE_W}}"
    try:
        pct = float(p) * 100
    except (TypeError, ValueError):
        return f"{'n/a':>{VALUE_W}}"
    if math.isnan(pct):
        return f"{'n/a':>{VALUE_W}}"
    if 0 < pct < 0.01:
        s = "<0.01%"
    else:
        s = f"{pct:.2f}%"
    return f"{s:>{VALUE_W}}"


def format_message(object_id, result, *, title="TEMPO", link=None, extra_text=None):
    leaf = result["leaf"]
    class_names = list(leaf["class_names"])
    probs_final = list(leaf["probs_final"])
    stds = list(leaf.get("prob_std_calibrated") or [None] * len(class_names))
    ranked = sorted(
        zip(class_names, probs_final, stds), key=lambda r: -float(r[1])
    )

    if link:
        header = f"*{title} — <{link}|{object_id}>*"
    else:
        header = f"*{title} — {object_id}*"
    lines = [header, "```"]
    for name, p, s in ranked:
        std_str = f"±{float(s) * 100:5.2f}%" if s is not None else ""
        lines.append(f"{name:<{LABEL_W}} {_fmt_pct(p)}  {std_str}")
    lines.append("```")

    decision = result.get("decision", {})
    fallback = decision.get("fallback") or {}
    leaf_kept = bool(leaf.get("kept"))
    if decision.get("abstain_completely"):
        lines.append("decision: *abstain*")
    elif fallback.get("label"):
        lines.append(
            f"decision: *{fallback['label']}* (level={fallback.get('level_name')}, leaf_kept={leaf_kept})"
        )

    sel_name = leaf.get("uncertainty_selected_name")
    sel_val = leaf.get("uncertainty_selected_value")
    if sel_name is not None and sel_val is not None:
        lines.append(f"uncertainty gate: {sel_name} = {float(sel_val):.4f}")

    ood = leaf.get("ood")
    if ood and ood.get("flag"):
        lines.append(f"OOD flag: yes (votes={ood.get('votes')})")

    if extra_text:
        lines.append(extra_text)
    return "\n".join(lines)


def generate_image(object_id, result, *, title="TEMPO", out_dir=None, font_size=12, taxonomy=None):
    out_dir = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
    out = out_dir / f"{object_id}_tempo_sunburst.png"
    plot_title = f"{title} — {object_id}"
    return save_sunburst(
        result,
        taxonomy or get_taxonomy(),
        out,
        title=plot_title,
        font_size=font_size,
    )


def post_to_slack(
    object_id,
    result,
    *,
    title="TEMPO",
    link=None,
    token=None,
    channel=None,
    token_env="SLACK_TEMPO_BOT_TOKEN",
    channel_env="SLACK_TEMPO_CHANNEL_ID",
    image_path=None,
    font_size=12,
    extra_text=None,
):
    """Upload a TEMPO summary image and probability block to Slack.

    Returns the Slack file id on success, or None if env isn't configured.
    """
    token = token or os.getenv(token_env)
    channel = channel or os.getenv(channel_env)
    if not token or not channel:
        logger.warning(
            "[%s] %s/%s not set, skipping post", object_id, token_env, channel_env
        )
        return None

    if image_path is None:
        image_path = generate_image(object_id, result, title=title, font_size=font_size)
    image_path = Path(image_path)
    message = format_message(
        object_id, result, title=title, link=link, extra_text=extra_text
    )

    size = image_path.stat().st_size
    r = requests.get(
        SLACK_GET_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={"filename": image_path.name, "length": size},
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    if not j.get("ok"):
        logger.error("[%s] Slack getUploadURLExternal failed: %s", object_id, j)
        return None
    upload_url = j["upload_url"]
    file_id = j["file_id"]

    with open(image_path, "rb") as f:
        r = requests.post(upload_url, files={"file": (image_path.name, f)}, timeout=60)
    r.raise_for_status()

    r = requests.post(
        SLACK_COMPLETE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json={
            "files": [{"id": file_id, "title": f"{object_id} {title}"}],
            "channel_id": channel,
            "initial_comment": message,
        },
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    if not j.get("ok"):
        logger.error("[%s] Slack completeUploadExternal failed: %s", object_id, j)
        return None
    return file_id
