"""
Bar chart of how different foods and risk factors change the risk of cancer.

Reads a CSV with these columns (one row per result):

    label    Name shown on the chart, e.g. "Pickled vegetables"
    risk     How many times as likely (1.28 = 28% more likely, 0.5 = half as likely).
             Leave blank if the study only gives a range (see low/high).
    low      Optional. Bottom of the range.
    high     Optional. Top of the range.
               - With a risk value: drawn as a thin whisker (e.g. a 95% confidence interval).
                 If the whisker crosses 1, the bar is grey: the result isn't statistically significant.
               - Without a risk value: drawn as a lighter bar covering the range (e.g. salt, 1.3-3.8).
    disease  Optional. The cancer the result refers to. Each one gets its own panel with a heading,
             in the order it first appears in the CSV.
    source   Short citation. Every different source is listed under the chart and shown on hover.

Usage:
    python risk_chart.py                                  # all rows -> risk_chart.html (+ .png)
    python risk_chart.py --disease "stomach cancer"       # only stomach cancer rows
    python risk_chart.py --disease "colorectal cancer,rectal cancer" -o dairy_chart
    python risk_chart.py my_data.csv -o chart             # custom input and output names

Needs:  pip install plotly pandas
For the PNG (for Substack or other places that only take images):  pip install kaleido
"""

import argparse
import math
import re
import textwrap

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Text you may want to edit -------------------------------------------------
TITLE = "How different foods and risk factors are linked to cancer risk"
SUBTITLE = "Compared with people who ate less or didn't have the risk factor. 1 = no change in risk."
X_AXIS_TITLE = "Times as likely to develop cancer"
ASTERISK_NOTE = ("* These results are for high intake. The study also included risk levels "
                 "for moderate intake.")   # shown only if a label ends with *
NOT_SIGNIFICANT_NOTE = ("Grey bars: not statistically significant.")

# ---- Colours and styling -------------------------------------------------------
HIGHER = "#e34948"      # red: higher risk
LOWER = "#2a78d6"       # blue: lower risk
NEUTRAL = "#a3a29b"     # grey: not statistically significant
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# ---- Size: lower these to make the chart shorter --------------------------------
ROW_HEIGHT = 40         # pixels per bar (wrapped two-line names need ~36+)
HEADING_GAP = 34        # pixels between sections (room for each cancer heading)
WIDTH = 640             # pixels; the title/notes wrap to fit this, so keep it near your blog column width
LABEL_WRAP = 22         # characters per line for the bar names on the left
X_MAX = None            # right end of the x axis, e.g. 3.5. Leave as None to work it out automatically
HEADROOM = 1.1          # automatic mode: room beyond the highest value for the value labels (1.1 = 10%)
WHISKER_LABEL_GAP = 2   # spaces between a whisker's end and its value label


def fmt(x):
    """1.28 -> '1.28', 0.50 -> '0.5', 3.80 -> '3.8'."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def load(path, diseases=None):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("risk", "low", "high"):
        if col not in df:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("source", "disease"):
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    if diseases:
        wanted = {d.strip().lower() for d in diseases.split(",")}
        df = df[df["disease"].str.lower().isin(wanted)]
        if df.empty:
            raise ValueError(f"No rows match --disease {diseases!r}")

    df["is_range"] = df["risk"].isna() & df["low"].notna() & df["high"].notna()
    bad = df["risk"].isna() & ~df["is_range"]
    if bad.any():
        raise ValueError("These rows need a risk value or both low and high: "
                         + ", ".join(df.loc[bad, "label"]))

    df["has_whisker"] = ~df["is_range"] & df["low"].notna() & df["high"].notna()
    df["not_sig"] = df["has_whisker"] & (df["low"] <= 1) & (df["high"] >= 1)
    # Value used for sorting and colour: the risk, or the middle of a range.
    df["mid"] = df["risk"].where(~df["is_range"], (df["low"] + df["high"]) / 2)
    df["text"] = [f"{fmt(r.low)}–{fmt(r.high)}×" if r.is_range else f"{fmt(r.risk)}×"
                  for r in df.itertuples()]

    # Panels in the order each disease first appears; highest risk at the top within each.
    order = {d: i for i, d in enumerate(dict.fromkeys(df["disease"]))}
    df["panel"] = df["disease"].map(order)

    # Keep related rows next to each other (e.g. "High salt intake" and "Moderate salt intake").
    # Rows are grouped by the optional 'group' column, or else by their label with any leading
    # "High/Moderate/Low/Medium" and trailing * removed. Each group sits where its highest
    # risk would put it, with its rows sorted highest to lowest inside it.
    if "group" in df:
        df["group"] = df["group"].fillna("").astype(str).str.strip()
    else:
        df["group"] = ""
    auto_key = (df["label"].str.rstrip("* ").str.lower()
                .str.replace(r"^(very high|high|moderate|medium|low|very low)\s+", "", regex=True))
    df["group"] = df["group"].where(df["group"] != "", auto_key)
    df["group_mid"] = df.groupby(["panel", "group"])["mid"].transform("max")
    return df.sort_values(["panel", "group_mid", "group", "mid"],
                          ascending=[True, False, True, False]).reset_index(drop=True)


def wrap(text, font_size):
    """Split text into lines that fit the chart width (Plotly doesn't wrap text itself)."""
    chars = int((WIDTH - 60) / (font_size * 0.6))   # allows for margins and wide letters
    return textwrap.wrap(text, chars) or [""]


def heading(disease):
    return disease[:1].upper() + disease[1:] if disease else ""


def add_row(fig, r, row):
    if r.not_sig:
        colour = NEUTRAL
    else:
        colour = HIGHER if r.mid >= 1 else LOWER

    if r.is_range:
        base, length, opacity = r.low, r.high - r.low, 0.55
        hover_value = f"{fmt(r.low)} to {fmt(r.high)} times as likely (range)"
    else:
        base, length, opacity = 1, r.risk - 1, 1
        hover_value = f"{fmt(r.risk)} times as likely"
        if r.has_whisker:
            hover_value += f" (range {fmt(r.low)}–{fmt(r.high)})"
        if r.not_sig:
            hover_value += "<br>Not statistically significant"
    if r.disease:
        hover_value += f"<br>{heading(r.disease)}"

    fig.add_trace(go.Bar(
        y=[r.label], x=[length], base=[base], orientation="h",
        marker=dict(color=colour, opacity=opacity, line=dict(width=0), cornerradius=4),
        width=0.62,
        text=[None if r.has_whisker else r.text], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK, size=13),
        customdata=[[hover_value, r.source]],
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>"
                      "<span style='color:#898781'>%{customdata[1]}</span><extra></extra>",
        showlegend=False,
    ), row=row, col=1)

    if r.has_whisker:
        # Whisker drawn as its own layer so it lines up with the risk value.
        fig.add_trace(go.Scatter(
            y=[r.label], x=[r.risk], mode="markers",
            marker=dict(size=1, opacity=0), hoverinfo="skip", showlegend=False,
            error_x=dict(type="data", symmetric=False,
                         array=[r.high - r.risk], arrayminus=[r.risk - r.low],
                         color=INK_2, thickness=1.5, width=6),
        ), row=row, col=1)
        # Value label sits past the end of the whisker so the two don't overlap.
        right = r.mid >= 1
        gap = "\u00a0" * WHISKER_LABEL_GAP   # non-breaking spaces (normal spaces get dropped)
        fig.add_trace(go.Scatter(
            y=[r.label], x=[r.high if right else r.low], mode="text",
            text=[gap + r.text if right else r.text + gap], textposition="middle right" if right else "middle left",
            textfont=dict(color=INK, size=13), cliponaxis=False,
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)


def build(df):
    panels = list(dict.fromkeys(df["disease"]))
    counts = [int((df["disease"] == d).sum()) for d in panels]
    titled = any(panels)

    plot_height = ROW_HEIGHT * len(df) + HEADING_GAP * (len(panels) - 1)
    fig = make_subplots(
        rows=len(panels), cols=1, shared_xaxes=True,
        row_heights=counts, vertical_spacing=HEADING_GAP / plot_height if len(panels) > 1 else 0,
        subplot_titles=[f"<b>{heading(d)}</b>" for d in panels] if titled else None,
    )
    for i, d in enumerate(panels, start=1):
        sub = df[df["disease"] == d]
        for r in sub.itertuples():
            add_row(fig, r, i)
        labels = list(sub["label"])[::-1]
        fig.update_yaxes(categoryorder="array", categoryarray=labels,
                         tickmode="array", tickvals=labels,
                         ticktext=["<br>".join(textwrap.wrap(l, LABEL_WRAP)) or l for l in labels],
                         row=i, col=1)

    # Panel headings: left-aligned above each panel.
    for a in fig.layout.annotations:
        a.update(x=0, xanchor="left", yshift=-2, font=dict(size=13, color=INK))

    # Axis range: start at 0, leave room on the right for the value labels.
    top = max(df["high"].max(skipna=True) if df["high"].notna().any() else 0,
              df["risk"].max(skipna=True) if df["risk"].notna().any() else 0, 1)
    x_max = X_MAX if X_MAX else math.ceil(top * HEADROOM * 2) / 2

    # "No change" line at 1 through every panel, with direction labels above the first.
    fig.add_vline(x=1, line=dict(color=INK_2, width=1.5, dash="dot"), row="all", col=1)
    label_y = dict(y=1, yref="paper", yanchor="bottom", yshift=22 if titled else 2, showarrow=False)
    fig.add_annotation(x=1, text="No change", font=dict(color=INK_2, size=11), **label_y)
    fig.add_annotation(x=1, xanchor="right", xshift=-30, text="← Lower risk",
                       font=dict(color=LOWER, size=11), **label_y)
    fig.add_annotation(x=1, xanchor="left", xshift=30, text="Higher risk →",
                       font=dict(color=HIGHER, size=11), **label_y)

    notes = []
    if df["label"].str.rstrip().str.endswith("*").any():
        notes.append(ASTERISK_NOTE)
    if df["not_sig"].any():
        notes.append(NOT_SIGNIFICANT_NOTE)
    sources = [s for s in dict.fromkeys(df["source"]) if s]
    if sources:
        if len(sources) == 1:
            notes.append("Source: " + sources[0])
        else:
            notes.append("Sources:")
            notes.extend(sources)   # one source per line
    note_lines = [line for n in notes for line in wrap(n, 11)]
    if note_lines:
        fig.add_annotation(x=0, xref="paper", xanchor="left", y=0, yref="paper", yanchor="top",
                           yshift=-46, align="left", showarrow=False,
                           text="<br>".join(note_lines), font=dict(color=MUTED, size=11))

    fig.update_xaxes(range=[0, x_max], dtick=0.5 if x_max <= 3 else 1, ticksuffix="×",
                     zeroline=False, showgrid=True, gridcolor=GRID, gridwidth=1,
                     showline=True, linecolor=BASELINE, tickfont=dict(color=MUTED))
    fig.update_xaxes(title=dict(text=X_AXIS_TITLE, font=dict(color=INK_2, size=12)),
                     row=len(panels), col=1)
    fig.update_yaxes(automargin=True, showgrid=False, ticks="", tickfont=dict(color=INK, size=11))

    subtitle_lines = wrap(SUBTITLE, 13)
    top_margin = 62 + 17 * len(subtitle_lines) + (26 if titled else 8)
    bottom_margin = 52 + 15 * len(note_lines)
    fig.update_layout(
        title=dict(text=f"<b>{TITLE}</b><br><span style='font-size:13px;color:{INK_2}'>"
                        + "<br>".join(subtitle_lines) + "</span>",
                   x=0, xanchor="left", y=1, yanchor="top", yref="container", pad=dict(t=14),
                   font=dict(size=17, color=INK)),
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        bargap=0.3,
        margin=dict(l=10, r=40, t=top_margin, b=bottom_margin),
        height=top_margin + bottom_margin + plot_height,
        width=WIDTH,
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK)),
    )
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="food_risk.csv", help="input CSV (default: food_risk.csv)")
    p.add_argument("-o", "--out", default="risk_chart", help="output file name without extension")
    p.add_argument("-d", "--disease", help="only include these diseases (comma-separated)")
    args = p.parse_args()

    fig = build(load(args.csv, args.disease))
    try:
        fig.write_image(f"{args.out}.png", scale=2)   # scale=2 keeps it sharp on high-res screens
        print(f"Saved {args.out}.png (image)")
    except Exception as e:  # kaleido missing or not working
        print(f"PNG not saved ({e.__class__.__name__}). Run 'pip install kaleido' to export images.")

    # Responsive version for the blog: fills whatever width the page gives it.
    height = int(fig.layout.height)          # natural height worked out from the number of bars
    fig.update_layout(autosize=True, width=None)   # width follows the page; height stays fixed
    fig.write_html(
        f"{args.out}.html",
        include_plotlyjs="cdn",
        config={"responsive": True},
        default_width="100%",
        default_height=f"{height}px",
    )
    print(f"Saved {args.out}.html (interactive)")
    print(f"Set the iframe height in your post to {height + 10}")

if __name__ == "__main__":
    main()
