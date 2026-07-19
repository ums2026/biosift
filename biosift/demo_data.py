from __future__ import annotations

import itertools

import pandas as pd

DEMO_PROTOCOL = """
Study title: MEK inhibitor response in patient-derived colorectal cancer organoids

Study design
Patient-derived colorectal cancer organoid cultures were stratified by KRAS
status as wild-type or mutant. Organoids received vehicle control or 100 nM
trametinib, a MEK inhibitor, and were collected at baseline, 24 hours, and
72 hours for transcriptomic profiling. The intended design contained four
biological replicates for every combination of KRAS status, treatment, and time
point. Samples were processed in two library-preparation batches, Batch 1 and
Batch 2. Investigators intended to compare treatment response while controlling
for genotype, collection time, and sequencing batch.

Sample metadata
Each sample has a unique sample identifier and a donor-derived organoid line
identifier. The metadata also records biological replicate, donor sex, drug
dose, processing batch, and a proposed machine-learning split. Samples from the
same donor-derived line must not appear in both training and test sets.

Expected factors
Organism: Homo sapiens
Tissue: colorectal tumor organoid
Condition: KRAS wild-type or KRAS mutant
Treatment: vehicle or trametinib
Dose: 0 or 100 nanomolar
Time points: baseline, 24 hours, 72 hours
Batches: Batch 1 and Batch 2
""".strip()


def _variant(value: str | int, variants: dict, index: int) -> str:
    options = variants[value]
    return options[index % len(options)]


def get_demo_dataframe() -> pd.DataFrame:
    conditions = ["wild_type", "mutant"]
    treatments = ["vehicle", "trametinib"]
    timepoints = [0, 24, 72]
    replicates = [1, 2, 3, 4]

    condition_variants = {
        "wild_type": ["WT", "KRAS-WT", "wild type"],
        "mutant": ["MUT", "KRAS-mut", "mutant"],
    }
    treatment_variants = {
        "vehicle": ["Veh", "vehicle", "Vehicle Control"],
        "trametinib": ["Tram", "trametinib", "MEKi"],
    }
    time_variants = {
        0: ["BL", "0h", "baseline"],
        24: ["D1", "24h", "Day1"],
        72: ["D3", "72h", "Day3"],
    }

    rows = []
    donor_counter = 1

    for condition, treatment, timepoint, replicate in itertools.product(
        conditions, treatments, timepoints, replicates
    ):
        # Plant one entirely missing design cell for the quality-control demo.
        if condition == "mutant" and treatment == "vehicle" and timepoint == 72:
            continue

        donor_id = f"ORG{donor_counter:02d}"
        donor_counter += 1
        sample_id = f"{donor_id}_{condition[:2].upper()}_{treatment[:4]}_{timepoint}h"

        # Strong treatment-batch confounding, with a few exceptions.
        if treatment == "trametinib":
            batch = "2" if replicate != 4 else "1"
        else:
            batch = "1" if replicate != 4 else "2"

        split = "train" if replicate in {1, 2, 3} else "test"

        rows.append(
            {
                "Sample": sample_id,
                "Donor": donor_id,
                "grp": _variant(condition, condition_variants, replicate + timepoint),
                "Tx": _variant(treatment, treatment_variants, replicate + timepoint),
                "tm": _variant(timepoint, time_variants, replicate),
                "lib": batch,
                "rep": f"R{replicate}",
                "set": split,
                "dose_nM": 100.0 if treatment == "trametinib" else 0.0,
                "sex": "F" if donor_counter % 2 == 0 else "M",
            }
        )

    df = pd.DataFrame(rows)

    # Plant a missing treatment label.
    df.loc[5, "Tx"] = None

    # Plant a duplicated sample identifier.
    df.loc[df.index[-1], "Sample"] = df.loc[df.index[-2], "Sample"]

    # Plant donor leakage by reusing a train donor in a test sample.
    train_donor = df.loc[df["set"] == "train", "Donor"].iloc[0]
    test_index = df.index[df["set"] == "test"][0]
    df.loc[test_index, "Donor"] = train_donor

    return df
