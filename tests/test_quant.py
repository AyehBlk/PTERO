"""Tests for ptero.quant -- multi-sample MaxQuant quant loading + display.

Synthetic-first (ROADMAP discipline): a tiny BOM-encoded proteinGroups.txt with
known values exercises 0->NaN, the Intensity>0 / LFQ=0 distinction, decoy /
contaminant / site filtering with per-reason counts, GN= gene parsing, explicit
group validation, and the plot builders (headless, via the Figure object API).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ptero.quant import (
    QuantData, load_quant, read_sample_names, suggest_groups_by_token,
    plot_protein_quant, plot_lfq_distributions, plot_detection_counts,
)

SAMPLES = ["S_A_1", "S_A_2", "S_B_1"]


def _row(id_, maj, fasta, rev, con, site, lfq, inten, score="100", rzr="4"):
    return {
        "id": id_,
        "Protein IDs": maj, "Majority protein IDs": maj, "Fasta headers": fasta,
        "Species": "Homo sapiens", "Q-value": "0", "Score": score,
        "Razor + unique peptides": rzr,
        "Reverse": rev, "Potential contaminant": con, "Only identified by site": site,
        "LFQ intensity S_A_1": lfq[0], "LFQ intensity S_A_2": lfq[1],
        "LFQ intensity S_B_1": lfq[2],
        "Intensity S_A_1": inten[0], "Intensity S_A_2": inten[1],
        "Intensity S_B_1": inten[2],
    }


def _write_protein_groups(path):
    rows = [
        _row(0, "sp|P1|P1_X", "sp|P1|P1_X X OS=Homo sapiens OX=9606 GN=Gene1 PE=1 SV=1",
             "", "", "", ("1000", "1100", "900"), ("2000", "2200", "1800")),
        # LFQ 0 in S_A_1 but Intensity > 0 there -> distinct state; S_B_1 both 0
        _row(1, "sp|P2|P2_X", "sp|P2|P2_X X OS=Homo sapiens OX=9606 GN=Gene2 PE=1 SV=1",
             "", "", "", ("0", "500", "0"), ("300", "900", "0")),
        _row(2, "CON__X", "", "", "+", "",                     # contaminant -> dropped
             ("10", "10", "10"), ("20", "20", "20")),
        _row(3, "REV__Y", "", "+", "", "",                     # reverse -> dropped
             ("1", "1", "1"), ("2", "2", "2")),
        _row(4, "sp|P5|P5_X", "sp|P5|P5_X X OS=Homo sapiens OX=9606 GN=Gene5 PE=1 SV=1",
             "", "", "+", ("7", "7", "7"), ("9", "9", "9")),   # only-by-site -> dropped
        # no GN= -> gene None, label falls back to leading accession P6
        _row(5, "sp|P6|P6_X;sp|P6b|P6b_X",
             "sp|P6|P6_X No gene OS=Homo sapiens OX=9606 PE=1 SV=1",
             "", "", "", ("800", "0", "700"), ("1600", "0", "1400")),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8-sig")  # BOM, like MaxQuant/Windows


@pytest.fixture()
def pg_path(tmp_path):
    p = tmp_path / "proteinGroups.txt"
    _write_protein_groups(p)
    return p


def test_read_sample_names(pg_path):
    assert read_sample_names(pg_path) == SAMPLES


def test_load_basic_shape_and_filtering(pg_path):
    qd = load_quant(pg_path)
    assert isinstance(qd, QuantData)
    assert qd.samples == SAMPLES
    assert qd.n_protein_groups_raw == 6
    assert list(qd.lfq.index) == [0, 1, 5]            # 2,3,4 dropped
    assert qd.n_filtered["potential_contaminant"] == 1
    assert qd.n_filtered["reverse"] == 1
    assert qd.n_filtered["only_identified_by_site"] == 1
    assert qd.n_filtered["rows_dropped"] == 3
    # raw-intensity columns aligned to the LFQ sample order
    assert list(qd.intensity.columns) == SAMPLES


def test_zero_becomes_nan_but_intensity_kept(pg_path):
    qd = load_quant(pg_path)
    assert np.isnan(qd.lfq.loc[1, "S_A_1"])           # LFQ 0 -> NaN
    assert qd.intensity.loc[1, "S_A_1"] == 300.0      # raw Intensity kept
    assert np.isnan(qd.lfq.loc[1, "S_B_1"])           # both 0
    assert np.isnan(qd.intensity.loc[1, "S_B_1"])
    assert qd.lfq.loc[0].notna().all()                # fully present


def test_gene_and_label_parsing(pg_path):
    qd = load_quant(pg_path)
    assert qd.proteins.loc[0, "gene"] == "Gene1"
    assert qd.proteins.loc[0, "label"] == "Gene1"
    assert qd.proteins.loc[5, "gene"] is None         # no GN=
    assert qd.proteins.loc[5, "label"] == "P6"        # leading accession


def test_keep_decoys_when_disabled(pg_path):
    qd = load_quant(pg_path, drop_decoys_contaminants=False)
    assert list(qd.lfq.index) == [0, 1, 2, 3, 4, 5]
    assert qd.n_filtered["rows_dropped"] == 0
    assert qd.n_filtered["reverse"] == 1              # still reported


def test_explicit_groups_validation(pg_path):
    groups = {"S_A_1": "A", "S_A_2": "A", "S_B_1": "B"}
    qd = load_quant(pg_path, groups=groups)
    assert qd.grouped
    assert qd.group_members() == {"A": ["S_A_1", "S_A_2"], "B": ["S_B_1"]}
    with pytest.raises(ValueError):                   # unmapped sample
        load_quant(pg_path, groups={"S_A_1": "A", "S_A_2": "A"})
    with pytest.raises(ValueError):                   # stray sample
        load_quant(pg_path, groups={**groups, "GHOST": "C"})


def test_suggest_groups_by_token():
    g = suggest_groups_by_token(SAMPLES, sep="_", index=1)
    assert g == {"S_A_1": "A", "S_A_2": "A", "S_B_1": "B"}
    assert suggest_groups_by_token(SAMPLES, sep="_", index=-2) == g
    with pytest.raises(ValueError):
        suggest_groups_by_token(["nodelim"], sep="_", index=1)


def test_to_long(pg_path):
    qd = load_quant(pg_path, groups={"S_A_1": "A", "S_A_2": "A", "S_B_1": "B"})
    long = qd.to_long()
    assert {"id", "sample", "group", "lfq", "intensity", "label", "gene"} <= set(long.columns)
    assert len(long) == qd.n_proteins * len(qd.samples)   # 3 * 3
    assert set(long["group"]) == {"A", "B"}


def test_plots_build_headless(pg_path):
    qd = load_quant(pg_path, groups={"S_A_1": "A", "S_A_2": "A", "S_B_1": "B"})
    f1 = plot_protein_quant(qd, 1)                    # id 1 has a missing sample
    f2 = plot_lfq_distributions(qd)
    f3 = plot_detection_counts(qd)
    for f in (f1, f2, f3):
        assert isinstance(f, Figure)
    qd2 = load_quant(pg_path)                         # ungrouped path
    assert isinstance(plot_lfq_distributions(qd2), Figure)
