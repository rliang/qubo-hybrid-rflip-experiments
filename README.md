## Prerequisites

- Packages: `sudo apt install g++ make curl`

## Compiling

```
make
```

## Running

Assuming `results.json` will be the file where experimental data is written to.

### Evaluation experiments

```sh
./main.out results.json eval
```

### LS experiments

```sh
./main.out results.json ls
./main.out results.json ls_count
```

### VNS experiments (figures)

```sh
./main.out results.json vns_figures
./main.out results.json vns_figures_count
```

### VNS experiments (tables)

```sh
./main.out results.json vns_tables_count
```

## Generating LaTeX figures and tables

Python 3 and the PyLatex package are required.

```sh
sudo apt install python3-pip
pip3 install --user pylatex
./genfigures.py results.json
./genfigures.py results.json
```