#!/usr/bin/env python3

import json
import sys


def percentage_change(x: float, y: float):
    return ((y - x) * 100) / x


with open(sys.argv[1]) as fp:
    FULL_DATA = json.load(fp)

for instance, f_prev in {
    "bqp50.1": 2098,
    "bqp50.2": 3702,
    "bqp50.3": 4626,
    "bqp50.4": 3544,
    "bqp50.5": 4012,
    "bqp50.6": 3693,
    "bqp50.7": 4520,
    "bqp50.8": 4216,
    "bqp50.9": 3780,
    "bqp50.10": 3507,
    "bqp100.1": 7970,
    "bqp100.2": 11036,
    "bqp100.3": 12723,
    "bqp100.4": 10368,
    "bqp100.5": 9083,
    "bqp100.6": 10210,
    "bqp100.7": 10125,
    "bqp100.8": 11435,
    "bqp100.9": 11455,
    "bqp100.10": 12565,
    "bqp250.1": 45607,
    "bqp250.2": 44810,
    "bqp250.3": 49037,
    "bqp250.4": 41274,
    "bqp250.5": 47961,
    "bqp250.6": 41014,
    "bqp250.7": 46757,
    "bqp250.8": 35726,
    "bqp250.9": 48916,
    "bqp250.10": 40442,
    "bqp500.1": 116586,
    "bqp500.2": 128223,
    "bqp500.3": 130812,
    "bqp500.4": 130097,
    "bqp500.5": 125487,
    "bqp500.6": 121772,
    "bqp500.7": 122201,
    "bqp500.8": 123559,
    "bqp500.9": 120798,
    "bqp500.10": 130619,
    "bqp1000.1": 371438,
    "bqp1000.2": 354932,
    "bqp1000.3": 371236,
    "bqp1000.4": 370675,
    "bqp1000.5": 352760,
    "bqp1000.6": 359629,
    "bqp1000.7": 371193,
    "bqp1000.8": 351994,
    "bqp1000.9": 349337,
    "bqp1000.10": 351415,
    "bqp2500.1": 1515944,
    "bqp2500.2": 1471392,
    "bqp2500.3": 1414192,
    "bqp2500.4": 1507701,
    "bqp2500.5": 1491816,
    "bqp2500.6": 1469162,
    "bqp2500.7": 1479040,
    "bqp2500.8": 1484199,
    "bqp2500.9": 1482413,
    "bqp2500.10": 1483355,
    "G1": 11624,
    "G2": 11620,
    "G3": 11622,
    "G4": 11646,
    "G5": 11631,
    "G6": 2178,
    "G7": 2003,
    "G8": 2003,
    "G9": 2048,
    "G10": 1994,
    "G11": 564,
    "G12": 556,
    "G13": 582,
    "G14": 3064,
    "G15": 3050,
    "G16": 3052,
    "G17": 3043,
    "G18": 988,
    "G19": 903,
    "G20": 941,
    "G21": 931,
    "G22": 13359,
    "G23": 13342,
    "G24": 13337,
    "G25": 13326,
    "G26": 13314,
    "G27": 3318,
    "G28": 3285,
    "G29": 3389,
    "G30": 3403,
    "G31": 3288,
    "G32": 1410,
    "G33": 1382,
    "G34": 1384,
    "G35": 7684,
    "G36": 7677,
    "G37": 7689,
    "G38": 7681,
    "G39": 2395,
    "G40": 2387,
    "G41": 2398,
    "G42": 2469,
    "G43": 6660,
    "G44": 6650,
    "G45": 6654,
    "G46": 6645,
    "G47": 6656,
    "G48": 6000,
    "G49": 6000,
    "G50": 5880,
    "G51": 3846,
    "G52": 3849,
    "G53": 3846,
    "G54": 3846,
}.items():
    n = next(
        (x["params"]["n"] for x in FULL_DATA if x["params"]["instance"] == instance),
        None,
    )
    if n is None:
        continue
    entries = []
    for r_max in range((n * 30) // 100, n, (n * 30) // 100):
        fx = next(
            (
                x["result"]["fx"]
                for x in FULL_DATA
                if x["params"]["exp"] == "vns_tables"
                and x["params"]["instance"] == instance
                and x["params"]["r_max"] == r_max
            ),
            None,
        )
        if fx is None:
            continue
        dt_b, dt_rv, dt_s = (
            next(
                (
                    x["result"]["dt"]
                    for x in FULL_DATA
                    if x["params"]["exp"] == "vns_tables"
                    and x["params"]["instance"] == instance
                    and x["params"]["r_max"] == r_max
                    and x["params"]["eval"] == algo_number
                ),
                None,
            )
            for algo_number in (0, 1, 2)
        )
        if dt_b is None or dt_rv is None or dt_s is None:
            continue
        entries.append(f_prev + fx)
        for t in (
            dt_b / 1e9,
            percentage_change(dt_b, dt_rv),
            percentage_change(dt_b, dt_s),
        ):
            entries.append("{0:.2f}".format(t))
    if len(entries) != 12:
        continue
    print(instance, f"n={n}")
    print(f_prev, *entries, sep="&", end="\\\\\n")
