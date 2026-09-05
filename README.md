# FH-RC-CPC

This repository contains the algorithms and public-data queries accompanying
the manuscript **Minimum-Cost Retention of Model Predictive Control Policies
under Changing Qualification Requirements**. It implements finite-registry
qualification, lossless reductions, complete optimum-class calculations, cost
and outcome-change certificates, and the dynamic-response retention query.

## Contents

```text
FH-RC-CPC/
├── src/fh_rc_cpc/
│   ├── qualification/       finite-registry construction and selection
│   ├── certificates/        cost, outcome, scope and reserve certificates
│   ├── response_query/      public pulse/recovery retention query
│   └── experiment_families/ analytic and seeded formulation comparisons
├── examples/                three executable paper examples
├── tests/                   focused scientific regression tests
├── pyproject.toml           package and dependency definition
└── requirements-lock.txt    versions used for release verification
```

Development notes, manuscript-production tools, plotting utilities and
industrial data-processing code are not part of this academic artifact.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

The versions used for release verification are listed in
`requirements-lock.txt`.

## Public data placement

Extract the code and data archives as sibling directories:

```text
parent_directory/
├── FH-RC-CPC/
└── Public_Result_Data/
```

The code never imports executable files from the data directory. Data files are
read through explicit paths and checked against their registered schemas and
hashes.

## Quick start

From the `FH-RC-CPC` directory after installation:

```bash
python examples/minimal_qualification.py
python examples/reproduce_certificates.py
```

Both commands use self-contained synthetic registries and print a JSON record
with `"status": "pass"` after their independent checks succeed.

## Reproducing the reported computations

| Manuscript result | Command |
|---|---|
| Exact selection and lossless signature evaluation | `python examples/minimal_qualification.py` |
| Cost-box and whole-row outcome certificates | `python examples/reproduce_certificates.py` |
| Analytic triangle, 12 scaling and 6 branching cases | `python examples/reproduce_formulation_comparison.py` |
| Public response-retention frontier and P0010 trajectories | `python -m fh_rc_cpc.response_query --data-dir ../Public_Result_Data` |
| Public qualification tables and scope optima | `python ../Public_Result_Data/verify_release.py` |

The formulation-comparison command runs every registered case and may take
several minutes. Runtime measurements depend on the machine and solver build;
the analytic identities, registered instances, feasibility, objectives and
certification status are checked independently of elapsed time.

A selected response tolerance can be queried with:

```bash
python -m fh_rc_cpc.response_query \
  --data-dir ../Public_Result_Data \
  --theta 0.027739077410182528
```

## Tests

With `Public_Result_Data` beside this directory:

```bash
python -m pytest -q
```

The four test files cover qualification and reductions, exact certificates,
the reported analytic/synthetic families, and the public response query.

## Data scope

The separate public archive contains the complete public benchmark records and
the response data used by the public retention query. Industrial historian
records are governed by site confidentiality and are available from the
corresponding author on reasonable request, subject to the applicable access
conditions.

## License and citation

The code is released under the MIT License. Citation metadata for the
accompanying manuscript are provided in `CITATION.cff`.
