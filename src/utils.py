from typing import Any


def format_tip(report: dict[str, Any]) -> str:
    """
    Format the personalized tips report into a readable string.

    Args:
        report (dict[str, Any]): The full report from generate_personalized_tips.

    Returns:
        str: A formatted multiline report including user and cluster refund info and top tips.
    """
    profile = report.get("user_profile", {})
    cluster_info = report.get("cluster_info", {})

    user_refund_amt = profile.get("refund_amount", 0.0)
    avg_cluster_refund_amt = cluster_info.get("avg_cluster_refund_amount", 0.0)

    lines = [
        f"Your refund amount: €{user_refund_amt:,.2f}",
        f"Peer cluster average refund amount: €{avg_cluster_refund_amt:,.2f}",
        "",
    ]

    tips = report.get("tips", [])
    for idx, tip in enumerate(tips, start=1):
        title = tip.get("title", "")
        message = tip.get("message", "").replace("\n", "\n    ")
        lines.append(f"{idx}. {title}")
        lines.append(f"    {message}")

    return "\n".join(lines)
