#!/usr/bin/env python3

import os
import json
import sys
from typing import List, Callable
from pylatex import Document, Command, TikZ, Axis, Plot, NoEscape


def percentage_change(x: float, y: float):
    return x / y


def percentage_usage(x: float, y: float):
    return (y * 100) / (x + y)


def groupby(func: Callable[[object], object], data: List[object]):
    return (
        (key, [x for x in data if func(x) == key])
        for key in sorted(set(map(func, data)))
    )


def create_figure(filename: str, options: List[str], *plots: Plot):
    doc = Document(documentclass="standalone", document_options="tikz,border=5pt")
    doc.append(Command("usepgfplotslibrary", "colorbrewer"))
    doc.append(Command("pgfplotsset", NoEscape("colormap/Paired-9")))
    with doc.create(TikZ()) as fig:
        fig_options = [
            "width=9cm",
            "height=5.5cm",
            "cycle list name=Paired-9",
            "legend style={font=\\footnotesize}",
            *options,
        ]
        with fig.create(Axis(NoEscape(",".join(fig_options)))) as axis:
            for plot in plots:
                axis.append(plot)
    doc.generate_tex(filename)
    os.system(f"latex {filename}.tex && dvips {filename}.dvi -o {filename}.eps")


with open(sys.argv[1]) as fp:
    FULL_DATA = json.load(fp)

for (instance, n), data in groupby(
    lambda x: (x["params"]["instance"], x["params"]["n"]),
    [x for x in FULL_DATA if x["params"]["exp"] == "eval"],
):
    for i, (n1, data) in enumerate(groupby(lambda x: x["params"]["n1"], data)):
        create_figure(
            f"eval-{instance}-{n1}".replace(".", "-"),
            [
                "xlabel={$r$ (\\% of $n$)}",
                "ylabel={speedup}",
                "xmin=1",
                "xmax=99",
                # "ymin=-120",
                # "ymax=420",
                # "ytick={-100, 0, 100, 200, 300, 400}",
                "legend cell align={left}",
                "legend columns=2",
                # "legend pos={north east}" if i < 2 else "legend pos={north west}",
                "legend pos={north east}"
            ],
            *[
                Plot(
                    name=Command("texttt", strategy_name),
                    coordinates=[
                        (
                            (r * 100 // n),
                            percentage_change(
                                next(
                                    x["result"]["avg"]
                                    for x in entries
                                    if x["params"]["eval"] == 0
                                ),
                                next(
                                    x["result"]["avg"]
                                    for x in entries
                                    if x["params"]["eval"] == strategy_number
                                ),
                            ),
                        )
                        for r, entries in groupby(lambda x: x["params"]["r"], data)
                    ],
                )
                for strategy_number, strategy_name in list(
                    enumerate(["B", "RV", "S", "A", "C", "M", "AC", "AM", "CM", "ACM"])
                )[2:]
            ],
            Plot(
                name=Command("texttt", "RV"),
                options="mark=none,dash dot,thick,color=gray",
                coordinates=[
                    (
                        (r * 100 // n),
                        percentage_change(
                            next(
                                x["result"]["avg"]
                                for x in entries
                                if x["params"]["eval"] == 0
                            ),
                            next(
                                x["result"]["avg"]
                                for x in entries
                                if x["params"]["eval"] == 1
                            ),
                        ),
                    )
                    for r, entries in groupby(lambda x: x["params"]["r"], data)
                ],
            ),
            Plot(
                options="mark=none,dashed,thick,color=red",
                coordinates=[(0, 1), (100, 1)],
            ),
        )

for (instance, n), data in groupby(
    lambda x: (x["params"]["instance"], x["params"]["n"]),
    [x for x in FULL_DATA if x["params"]["exp"] == "ls"],
):
    create_figure(
        f"ls-{instance}".replace(".", "-"),
        [
            "xlabel={$r$ (\\% of $n$)}",
            "ylabel={percentage change}",
            "xmin=1",
            "xmax=99",
            # "ymin=-120",
            # "ymax=120",
            # "ytick={-100, -50, 0, 50, 100}",
            "legend cell align={left}",
            "legend columns=3",
            # "legend pos={south east}",
            "legend pos={north east}",
        ],
        *[
            Plot(
                name=Command("texttt", strategy_name),
                coordinates=[
                    (
                        (r * 100 // n),
                        percentage_change(
                            next(
                                x["result"]["dt"]
                                for x in data
                                if x["params"]["eval"] == 0
                            ),
                            next(
                                x["result"]["dt"]
                                for x in data
                                if x["params"]["eval"] == strategy_number
                            ),
                        ),
                    )
                    for r, data in groupby(lambda x: x["params"]["r"], data)
                ],
            )
            for strategy_number, strategy_name in list(
                enumerate(["B", "RV", "S", "A", "C", "M", "AC", "AM", "CM", "ACM"])
            )[2:]
        ],
        Plot(
            name=Command("texttt", "RV"),
            options="mark=none,dash dot,thick,color=gray",
            coordinates=[
                (
                    (r * 100 // n),
                    percentage_change(
                        next(
                            x["result"]["dt"] for x in data if x["params"]["eval"] == 0
                        ),
                        next(
                            x["result"]["dt"] for x in data if x["params"]["eval"] == 1
                        ),
                    ),
                )
                for r, data in groupby(lambda x: x["params"]["r"], data)
            ],
        ),
        Plot(
            options="mark=none,dashed,thick,color=red", coordinates=[(0, 1), (100, 1)],
        ),
    )

for (instance, n), data in groupby(
    lambda x: (x["params"]["instance"], x["params"]["n"]),
    [x for x in FULL_DATA if x["params"]["exp"] == "ls_count"],
):
    create_figure(
        f"ls-count-{instance}".replace(".", "-"),
        [
            "xlabel={$r$ (\\% of $n$)}",
            "ylabel={\\% \\textit{r-flip-rv} usage}",
            "xmin=1",
            "xmax=99",
            "ytick={0, 25, 50, 75, 100}",
            "legend cell align={left}",
            "legend columns=2",
            "legend pos={north east}",
        ],
        *[
            Plot(
                name=Command("texttt", strategy_name),
                coordinates=[
                    (
                        (r * 100 // n),
                        percentage_usage(
                            next(
                                x["result"]["basics"]
                                for x in data
                                if x["params"]["eval"] == strategy_number
                            ),
                            next(
                                x["result"]["deltas"]
                                for x in data
                                if x["params"]["eval"] == strategy_number
                            ),
                        ),
                    )
                    for r, data in groupby(lambda x: x["params"]["r"], data)
                ],
            )
            for strategy_number, strategy_name in list(
                enumerate(["B", "RV", "S", "A", "C", "M", "AC", "AM", "CM", "ACM"])
            )[2:]
        ],
    )


for (instance, n), data in groupby(
    lambda x: (x["params"]["instance"], x["params"]["n"]),
    [x for x in FULL_DATA if x["params"]["exp"] == "vns_figures"],
):
    create_figure(
        f"vns-{instance}".replace(".", "-"),
        [
            "xlabel={$r$ (\\% of $n$)}",
            "ylabel={speedup}",
            "xmin=1",
            "xmax=99",
            # "ymin=-120",
            # "ymax=120",
            # "ytick={-100, -50, 0, 50, 100}",
            "legend cell align={left}",
            "legend columns=3",
            # "legend pos={north west}",
            "legend pos={north east}",
        ],
        *[
            Plot(
                name=Command("texttt", strategy_name),
                coordinates=[
                    (
                        (r * 100 // n),
                        percentage_change(
                            next(
                                x["result"]["dt"]
                                for x in data
                                if x["params"]["eval"] == 0
                            ),
                            next(
                                x["result"]["dt"]
                                for x in data
                                if x["params"]["eval"] == strategy_number
                            ),
                        ),
                    )
                    for r, data in groupby(lambda x: x["params"]["r_max"], data)
                ],
            )
            for strategy_number, strategy_name in list(
                enumerate(["B", "RV", "S", "A", "C", "M", "AC", "AM", "CM", "ACM"])
            )[2:]
        ],
        Plot(
            name=Command("texttt", "RV"),
            options="mark=none,dash dot,thick,color=gray",
            coordinates=[
                (
                    (r * 100 // n),
                    percentage_change(
                        next(
                            x["result"]["dt"] for x in data if x["params"]["eval"] == 0
                        ),
                        next(
                            x["result"]["dt"] for x in data if x["params"]["eval"] == 1
                        ),
                    ),
                )
                for r, data in groupby(lambda x: x["params"]["r_max"], data)
            ],
        ),
        Plot(
            options="mark=none,dashed,thick,color=red", coordinates=[(0, 1), (100, 1)],
        ),
    )

for (instance, n), data in groupby(
    lambda x: (x["params"]["instance"], x["params"]["n"]),
    [x for x in FULL_DATA if x["params"]["exp"] == "vns_figures_count"],
):
    create_figure(
        f"vns-count-{instance}".replace(".", "-"),
        [
            "xlabel={$r$ (\\% of $n$)}",
            "ylabel={\\% \\textit{r-flip-rv} usage}",
            "xmin=1",
            "xmax=99",
            "ymin=45",
            "ymax=105",
            "ytick={0, 25, 50, 75, 100}",
            "legend cell align={left}",
            "legend columns=2",
            "legend pos={south west}",
        ],
        *[
            Plot(
                name=Command("texttt", strategy_name),
                coordinates=[
                    (
                        (r * 100 // n),
                        percentage_usage(
                            next(
                                x["result"]["basics"]
                                for x in data
                                if x["params"]["eval"] == strategy_number
                            ),
                            next(
                                x["result"]["deltas"]
                                for x in data
                                if x["params"]["eval"] == strategy_number
                            ),
                        ),
                    )
                    for r, data in groupby(lambda x: x["params"]["r_max"], data)
                ],
            )
            for strategy_number, strategy_name in list(
                enumerate(["B", "RV", "S", "A", "C", "M", "AC", "AM", "CM", "ACM"])
            )[2:]
        ],
    )


os.system("rm *.aux *.dvi *.log")