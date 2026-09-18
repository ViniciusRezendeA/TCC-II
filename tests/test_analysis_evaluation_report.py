from __future__ import annotations

import pandas as pd
import pytest

from scripts.analysis_evaluation_report import (
    _benjamini_hochberg,
    tool_key_for,
    wilcoxon_por_componente,
)


def test_tool_key_for_disambiguates_colliding_tool_uid():
    """Two distinct tools sharing one tool_uid (lowlevel SDK pattern collision, see
    pipeline/run_step3.py::tool_uid_for) must produce different keys once tool.name differs.
    """
    shared_uid = "acme/x::handle_list_tools::server.py:10"
    record_a = {"tool_uid": shared_uid, "tool": {"name": "search_models"}}
    record_b = {"tool_uid": shared_uid, "tool": {"name": "search_datasets"}}

    assert tool_key_for(record_a) != tool_key_for(record_b)


def test_tool_key_for_agrees_across_pre_and_post_fix_vintages():
    """pipeline/run_step3.py::tool_uid_for() now bakes ::{name} into tool_uid itself for
    lowlevel patterns (evaluations collected after the fix); evaluations collected before it
    still have the un-suffixed tool_uid. The same real tool must land on the same key
    regardless of which vintage produced the record, or a tool evaluated by one judge before
    the fix and another judge after it would wrongly look like two different tools.
    """
    pre_fix = {"tool_uid": "acme/x::handle_list_tools::server.py:10", "tool": {"name": "search_datasets"}}
    post_fix = {"tool_uid": "acme/x::handle_list_tools::server.py:10::search_datasets", "tool": {"name": "search_datasets"}}

    assert tool_key_for(pre_fix) == tool_key_for(post_fix)


def test_benjamini_hochberg_matches_known_reference():
    # 5 p-valores, m=5: q(i) = min_{j>=i} (m/j * p(j)) sobre os p-valores ordenados
    # [0.01, 0.03, 0.04, 0.20, 0.50] -> raw [0.05, 0.075, 0.0667, 0.25, 0.50] -> step-up
    # (cummin de trás pra frente) [0.05, 0.0667, 0.0667, 0.25, 0.50], calculado à mão.
    pvalues = pd.Series([0.01, 0.04, 0.03, 0.20, 0.50])

    adjusted = _benjamini_hochberg(pvalues)

    expected = pd.Series([0.05, 2 / 30, 2 / 30, 0.25, 0.50])
    pd.testing.assert_series_equal(adjusted, expected, check_names=False)


def test_benjamini_hochberg_ignores_nan_and_preserves_index():
    pvalues = pd.Series([0.01, None, 0.02], index=["a", "b", "c"])

    adjusted = _benjamini_hochberg(pvalues)

    assert adjusted["b"] != adjusted["b"]  # NaN stays NaN
    assert adjusted["a"] == pytest.approx(0.02)
    assert adjusted["c"] == pytest.approx(0.02)


def _long_df_row(juiz, componente, tool_uid, cenario, nota):
    return {"juiz": juiz, "componente": componente, "tool_uid": tool_uid, "cenario": cenario, "nota": nota}


def test_wilcoxon_por_componente_counts_ties_and_pairs():
    """4 tools, mesmo juiz/componente: 2 empatadas (diferença 0), 2 com with_source > 0
    (diferença consistentemente positiva) -- n_pareado=4, n_empates=2, n_efetivo_teste=2."""
    rows = []
    for i, (desc, src) in enumerate([(3, 3), (2, 2), (1, 4), (2, 5)]):
        rows.append(_long_df_row("j", "purpose", f"tool_{i}", "description_only", desc))
        rows.append(_long_df_row("j", "purpose", f"tool_{i}", "with_source", src))
    long_df = pd.DataFrame(rows)

    result = wilcoxon_por_componente(long_df)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_pareado"] == 4
    assert row["n_empates"] == 2
    assert row["n_efetivo_teste"] == 2
    assert row["mediana_diferenca"] == 1.5  # medians of [0, 0, 3, 3]


def test_wilcoxon_por_componente_no_test_when_all_tied():
    rows = []
    for i in range(3):
        rows.append(_long_df_row("j", "purpose", f"tool_{i}", "description_only", 3))
        rows.append(_long_df_row("j", "purpose", f"tool_{i}", "with_source", 3))
    long_df = pd.DataFrame(rows)

    result = wilcoxon_por_componente(long_df)

    row = result.iloc[0]
    assert row["n_efetivo_teste"] == 0
    assert row["estatistica_w"] is None
    assert pd.isna(row["p_valor"])


def test_wilcoxon_por_componente_bh_correction_is_scoped_per_judge():
    """A família de comparações múltiplas é por juiz (6 componentes), não misturada entre
    juízes -- juiz_b não deve influenciar a correção aplicada às linhas do juiz_a."""
    rows = []
    for componente, (desc, src) in {
        "purpose": (1, 5),
        "guidelines": (1, 1),
    }.items():
        for tool_uid, delta in [("t0", 0), ("t1", 1), ("t2", 2)]:
            rows.append(_long_df_row("juiz_a", componente, tool_uid, "description_only", desc))
            rows.append(_long_df_row("juiz_a", componente, tool_uid, "with_source", src + delta))
    # juiz_b só tem um componente -- não deve afetar a correção de 6 (na prática, 2) do juiz_a.
    rows.append(_long_df_row("juiz_b", "purpose", "t0", "description_only", 1))
    rows.append(_long_df_row("juiz_b", "purpose", "t0", "with_source", 1))
    long_df = pd.DataFrame(rows)

    result = wilcoxon_por_componente(long_df)

    juiz_a = result[result["juiz"] == "juiz_a"].set_index("componente")
    # BH de 2 p-valores dentro de juiz_a, calculado independente do que juiz_b tem.
    manual_bh = _benjamini_hochberg(juiz_a["p_valor"])
    pd.testing.assert_series_equal(juiz_a["p_valor_bh"], manual_bh.round(4), check_names=False)
