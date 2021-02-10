"""Plot log results from Fluent."""

from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def read_logs_from(file_: str) -> Tuple[list, list]:
    """Read log without sim init stuff from Fluent run."""
    start = ["iter", "continuity", "x-velocity", "y-velocity", "z-velocity", "k", "epsilon", "time/iter"]
    pass_start = False
    log = []
    with open(file_, "r") as iofile:
        for line in iofile:
            if line.split() == start:
                pass_start = True
            if pass_start and (line != "\n"):
                log.append(line.split()[:-2])
    return log, start[:-1]


def parse(log: list) -> list:
    """Parse relevant info in log file and return data."""
    data = []
    for line in log:
        try:
            data.append([float(x) for x in line])
        except ValueError:
            pass
    return data


def plot(data: list, header: list) -> None:
    """Generate plots from Fluent logs."""
    df = pd.DataFrame(data)
    df.columns = header
    df.plot(x="iter", subplots=True, logy=True)
    plt.show()
    # plt.savefig("/tmp/prashant.svg")


def main():
    """Main plotting pipeline."""
    log, header = read_logs_from("./outputfile")
    data = parse(log)
    plot(data, header)


if __name__ == "__main__":
    main()
