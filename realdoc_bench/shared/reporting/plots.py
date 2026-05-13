"""Plot helpers shared by layout + extraction reports."""

from __future__ import annotations

from pathlib import Path

from realdoc_bench.shared.reporting.pareto import ParetoPoint, extract


def write_pareto_plot(
    points: list[tuple[str, float, float]],
    out_path: Path,
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> list[ParetoPoint]:
    """Render a scatter plot with the Pareto frontier highlighted.

    Returns the annotated point list (with on_frontier flags) so callers can also
    persist a JSON sidecar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    annotated = extract(points)
    frontier_pts = sorted(
        (p for p in annotated if p.on_frontier),
        key=lambda p: p.x,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        [p.x for p in annotated if not p.on_frontier],
        [p.y for p in annotated if not p.on_frontier],
        color="#bdbdbd",
        s=40,
        label="dominated",
    )
    ax.scatter(
        [p.x for p in frontier_pts],
        [p.y for p in frontier_pts],
        color="#1976d2",
        s=80,
        label="pareto frontier",
        zorder=3,
    )
    if len(frontier_pts) >= 2:
        ax.plot(
            [p.x for p in frontier_pts],
            [p.y for p in frontier_pts],
            color="#1976d2",
            alpha=0.5,
            zorder=2,
        )
    for p in annotated:
        ax.annotate(p.label, (p.x, p.y), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return annotated
