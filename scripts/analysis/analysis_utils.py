"""
Shared helpers for the modernisation analysis scripts that
do the following:

* check that required input tables exist;
* create the standard tables and figures folders;
* apply the shared Matplotlib visual style;
* add a title, a short description and a measure line;
* saving each figure as both PNG and SVG.

The default output is in English. I am attempting to alllow Spanish 
as an option. 

To produce Spanish figures as well, we first add the Spanish chart titles,
descriptions, measure lines and axis labels to each analysis script. Then
we change OUTPUT_LANGUAGES from ("en",) to ("en", "es").
"""

# ---------------------------------------------------------------------------
# Import the required modules
# ---------------------------------------------------------------------------

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


# ---------------------------------------------------------------------------
# Set the languages to generate
# ---------------------------------------------------------------------------

# Use ("en", "es") when Spanish chart text has been supplied. This will 
# just change any text in visualisations to Spanish.
OUTPUT_LANGUAGES = ("en",)


# ---------------------------------------------------------------------------
# Shared palette across all of the vis
# ---------------------------------------------------------------------------

COLOURS = {
    "background": "#F3EFE7",
    "panel": "#FFFDF8",
    "text": "#24211D",
    "muted_text": "#746D63",
    "grid": "#D8D0C4",
    "gold": "#B88A2D",
    "gold_light": "#E7D39B",
    "model_teal": "#1F6F78",
    "model_terracotta": "#B6533C",
    "missing": "#8A8177",
    "excluded": "#B9B1A6",
}

# Palette when showing all 6 error categories
ERROR_CATEGORY_COLOURS = (
    "#3E7185",  # muted blue
    "#B9633F",  # terracotta
    "#647A52",  # olive green
    "#76658A",  # muted violet
    "#A35E70",  # antique rose
    "#B58A2A",  # ochre
)


def choose_font(preferred_font, fallback_font):
    """Use a preferred installed font, otherwise return a safe fallback."""

    try:
        font_manager.findfont(preferred_font, fallback_to_default = False)
        return preferred_font
    except ValueError:
        return fallback_font


TITLE_FONT = choose_font("Source Serif 4", "DejaVu Serif")
BODY_FONT = choose_font("IBM Plex Sans", "DejaVu Sans")
MONO_FONT = choose_font("IBM Plex Mono", "DejaVu Sans Mono")


def apply_plot_style():
    """Apply the shared style to Matplotlib figures."""

    plt.rcParams.update(
        {
            "figure.facecolor": COLOURS["background"],
            "savefig.facecolor": COLOURS["background"],
            "axes.facecolor": COLOURS["panel"],
            "axes.edgecolor": COLOURS["grid"],
            "axes.labelcolor": COLOURS["text"],
            "axes.titlecolor": COLOURS["text"],
            "text.color": COLOURS["text"],
            "xtick.color": COLOURS["muted_text"],
            "ytick.color": COLOURS["muted_text"],
            "grid.color": COLOURS["grid"],
            "grid.linewidth": 0.8,
            "font.family": BODY_FONT,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def check_required_files(required_files, preceding_command = None):
    """Stop if one or more input tables are missing.

    Parameters
    ----------
    required_files : iterable of pathlib.Path
        Files that must exist before the analysis can run.
    preceding_command : str, optional
        Command that the user should run to create the missing inputs.
    """

    missing_files = [path for path in required_files if not path.is_file()]

    if not missing_files:
        return

    missing_list = "\n".join(f"- {path}" for path in missing_files)
    message = f"The following required files were not found:\n{missing_list}"

    if preceding_command:
        message += f"\n\nRun this first:\n{preceding_command}"

    raise SystemExit(message)


def create_output_folders(output_directory):
    """Create and return the standard tables and figures directories."""

    tables_directory = output_directory / "tables"
    figures_directory = output_directory / "figures"

    tables_directory.mkdir(parents = True, exist_ok = True)
    figures_directory.mkdir(parents = True, exist_ok = True)

    return tables_directory, figures_directory


def add_chart_header(fig, title, description, measure = None):
    """Add the question the vis is trying to answer and description of 
    what plotted values represent.

    * Title: states the question
    * Description explains what the figure is intended to show 
    * Measure (optional): what has been counted or calculated.
    """

    fig.text(
        0.08,
        0.965,
        fill(title, width = 72),
        fontsize=18,
        fontweight = "bold",
        fontfamily = TITLE_FONT,
        color = COLOURS["text"],
        va = "top",
    )

    fig.text(
        0.08,
        0.885,
        fill(description, width = 105),
        fontsize = 10.5,
        color = COLOURS["text"],
        va = "top",
    )

    if measure:
        fig.text(
            0.08,
            0.825,
            fill(measure, width = 110),
            fontsize = 9.5,
            fontstyle = "italic",
            color = COLOURS["muted_text"],
            va = "top",
        )

    fig.add_artist(
        plt.Line2D(
            [0.08, 0.92],
            [0.785, 0.785],
            transform = fig.transFigure,
            color = COLOURS["gold"],
            linewidth = 1.6,
        )
    )


def add_figure_note(fig, note):
    """Adds a small methodological note below a figure."""

    fig.text(
        0.08,
        0.025,
        fill(note, width = 125),
        fontsize = 8.5,
        color = COLOURS["muted_text"],
        va = "bottom",
    )


def save_figure(fig, output_directory, filename_stem):
    """Save one figure as a PNG for sharing and a scalable SVG"""

    output_directory.mkdir(parents = True, exist_ok = True)

    fig.savefig(
        output_directory / f"{filename_stem}.png",
        dpi = 300,
        bbox_inches = "tight",
        facecolor=fig.get_facecolor(),
    )

    fig.savefig(
        output_directory / f"{filename_stem}.svg",
        bbox_inches = "tight",
        facecolor = fig.get_facecolor(),
    )

    plt.close(fig)


def save_table(dataframe, output_path):
    """Saves an CSV (without a Pandas index column)"""

    output_path.parent.mkdir(parents = True, exist_ok = True)
    dataframe.to_csv(output_path, index = False, encoding = "utf-8-sig")


def coerce_boolean(series):
    """Convert common CSV boolean representations to Pandas nullable Boolean."""

    if str(series.dtype) == "boolean":
        return series

    value_map = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
    }

    converted = series.map(value_map)

    unrecognised = series.notna() & converted.isna()
    if unrecognised.any():
        values = sorted(series.loc[unrecognised].astype(str).unique())
        raise ValueError(
            f"Boolean column contains unrecognised values: {values}."
        )

    return converted.astype("boolean")


def model_colour_map(model_names):
    """Return stable colours for the two model names found in the data."""

    names = [str(name) for name in pd.Series(model_names).dropna().unique()]

    if len(names) > 2:
        raise ValueError(
            "The analysis theme currently expects no more than two models. "
            f"Found: {sorted(names)}."
        )

    colours = {}
    unassigned = []

    for name in names:
        lower_name = name.lower()
        if "llama" in lower_name:
            colours[name] = COLOURS["model_teal"]
        elif "gpt" in lower_name or "oss" in lower_name:
            colours[name] = COLOURS["model_terracotta"]
        else:
            unassigned.append(name)

    available_colours = [
        colour
        for colour in (
            COLOURS["model_teal"],
            COLOURS["model_terracotta"],
        )
        if colour not in colours.values()
    ]

    for name, colour in zip(sorted(unassigned), available_colours):
        colours[name] = colour

    return colours

