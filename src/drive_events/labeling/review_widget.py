"""Notebook review tool (works in Jupyter and Google Colab).

Shows one event card at a time with Accept / Reject / Relabel buttons. Every click
is written straight to the :class:`LabelStore` CSV, so a disconnect loses nothing -
put the CSV on Google Drive when working in Colab.

Usage::

    from drive_events.labeling import LabelStore, review_events
    store = LabelStore("/content/drive/MyDrive/drive_events/labels.csv")
    review_events(events, store, cards_dir="outputs/kitti/cards", reviewer="sandra")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drive_events.labeling.store import LabelStore


def review_events(
    events: pd.DataFrame,
    store: LabelStore,
    cards_dir: str | Path,
    reviewer: str = "",
    types: list[str] | None = None,
):
    """Display the review widget for all events without a decision yet."""
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The review widget needs: pip install ipywidgets") from exc

    cards_dir = Path(cards_dir)
    queue = store.pending(events)
    queue = queue[[(cards_dir / f"{eid}.png").exists() for eid in queue["event_id"]]]
    queue = queue.reset_index(drop=True)
    all_types = types or sorted(set(events["type"]) | {"other"})

    image = w.Image(format="png", layout=w.Layout(width="100%"))
    header = w.HTML()
    comment = w.Text(placeholder="optional comment", layout=w.Layout(width="40%"))
    relabel_to = w.Dropdown(options=all_types, description="relabel to",
                            layout=w.Layout(width="25%"))
    b_accept = w.Button(description="Accept", button_style="success", icon="check")
    b_reject = w.Button(description="Reject", button_style="danger", icon="times")
    b_relabel = w.Button(description="Relabel", button_style="warning", icon="tag")
    b_skip = w.Button(description="Skip", icon="forward")
    state = {"i": 0}

    def show() -> None:
        i = state["i"]
        if i >= len(queue):
            header.value = f"<b>Done.</b> {len(store)} events reviewed in total."
            image.value = b""
            for b in (b_accept, b_reject, b_relabel, b_skip):
                b.disabled = True
            return
        ev = queue.iloc[i]
        header.value = (f"<b>{i + 1} / {len(queue)}</b> &nbsp; {ev.event_id} &nbsp; "
                        f"rule says <b>{ev.type}</b> ({ev.severity})")
        image.value = (cards_dir / f"{ev.event_id}.png").read_bytes()
        comment.value = ""

    def decide(decision: str) -> None:
        ev = queue.iloc[state["i"]]
        store.record(ev.event_id, ev.sequence, ev.type, decision,
                     final_type=relabel_to.value if decision == "relabel" else None,
                     reviewer=reviewer, comment=comment.value)
        state["i"] += 1
        show()

    b_accept.on_click(lambda _: decide("accept"))
    b_reject.on_click(lambda _: decide("reject"))
    b_relabel.on_click(lambda _: decide("relabel"))
    b_skip.on_click(lambda _: (state.__setitem__("i", state["i"] + 1), show()))

    ui = w.VBox([header, image,
                 w.HBox([b_accept, b_reject, relabel_to, b_relabel, b_skip]), comment])
    show()
    display(ui)
    return ui
