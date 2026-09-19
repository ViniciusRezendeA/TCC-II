from __future__ import annotations

import pandas as pd
import pytest

from scripts.analysis_evaluation_report import (
    MOTIVO_FALLBACK,
    _benjamini_hochberg,
    _delta_custo_com_codigo,
    _deepseek_em_horario_peak_utc,
    classificar_motivos,
    custo_latencia_por_juiz_e_cenario,
    custo_real_por_juiz,
    custo_real_usd,
    motivos_por_divergencia,
    registros_versao_ativa,
    resumo_motivos_por_componente,
    tool_key_for,
    veredito_custo_beneficio,
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


def test_registros_versao_ativa_keeps_only_latest_version():
    old = {"prompt_version": "v3"}
    new_a = {"prompt_version": "v4"}
    new_b = {"prompt_version": "v4"}

    assert registros_versao_ativa([old, new_a, new_b]) == [new_a, new_b]


def test_registros_versao_ativa_empty_input():
    assert registros_versao_ativa([]) == []


def test_classificar_motivos_matches_known_keyword():
    assert classificar_motivos("The parameter type is not documented.") == ["parametro"]


def test_classificar_motivos_matches_multiple_categories():
    """Uma justificativa pode casar mais de uma categoria -- é uma junção por palavra-chave,
    não uma classificação exclusiva (ver docstring de classificar_motivos())."""
    assert set(classificar_motivos("The parameter is missing from the description.")) == {"parametro", "omissao"}


def test_classificar_motivos_falls_back_when_nothing_matches():
    assert classificar_motivos("Looks fine overall.") == [MOTIVO_FALLBACK]


def test_classificar_motivos_is_case_insensitive():
    assert classificar_motivos("CONTRADICTS the implementation") == ["contradicao"]


# --- Fixtures de registros brutos (schema de data/evaluations/{judge}.jsonl) -----------------

_ALL_COMPONENTS = ["purpose", "guidelines", "limitations", "parameter_explanation", "length_completeness", "examples"]


def _scores(overrides: dict[str, dict] | None = None) -> dict:
    """Um registro bruto precisa dos 6 componentes (scores_long() itera RUBRIC_COMPONENTS sem
    checar presença) -- os não sobrescritos ficam com nota 3 idêntica nos dois cenários, o que
    os torna empatados (ver empate em migracao_quartil_por_tool()) e portanto invisíveis para
    qualquer análise de divergência, sem precisar excluí-los explicitamente em cada teste.
    """
    base = {c: {"score": 3, "reasoning": "n/a"} for c in _ALL_COMPONENTS}
    base.update(overrides or {})
    return base


def _record(*, tool_uid, cenario, scores, juiz="j", status="ok", tool_name="tool",
            prompt_version="v4", input_tokens=100, output_tokens=50, latency_ms=1000.0):
    return {
        "tool_uid": tool_uid,
        "tool": {"name": tool_name, "qualified_name": tool_name},
        "repo": {"name_with_owner": "repo/x", "primary_language": "Python"},
        "judge": {"id": juiz, "provider": "test"},
        "scenario": cenario,
        "status": status,
        "scores": scores,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cache_read_input_tokens": 0},
        "latency_ms": latency_ms,
        "prompt_version": prompt_version,
    }


def test_motivos_por_divergencia_only_includes_diverging_pairs_and_classifies_with_source_reasoning():
    """4 tools, mesmo juiz/componente, notas de description_only distintas 1..4 (ranks exatos
    em quartis 1..4 sem empate) -- só t0 migra >=2 faixas (Q1->Q4); t1/t2/t3 migram só 1 faixa
    cada (não contam como divergência, ver MUDANCA_MINIMA_QUARTIS)."""
    desc_scores = {"t0": 1, "t1": 2, "t2": 3, "t3": 4}
    src_scores = {"t0": 5, "t1": 2, "t2": 3, "t3": 4}
    src_reasoning = {
        "t0": "The parameter type is missing from the description.",
        "t1": "fine", "t2": "fine", "t3": "fine",
    }
    records = []
    for tool_uid, desc in desc_scores.items():
        records.append(_record(tool_uid=tool_uid, cenario="description_only", scores=_scores({"purpose": {"score": desc, "reasoning": "n/a"}})))
        records.append(_record(tool_uid=tool_uid, cenario="with_source", scores=_scores({"purpose": {"score": src_scores[tool_uid], "reasoning": src_reasoning[tool_uid]}})))

    result = motivos_por_divergencia(records)

    purpose_rows = result[result["componente"] == "purpose"]
    assert set(purpose_rows["tool_uid"]) == {"t0::tool"}
    assert set(purpose_rows["motivo"]) == {"parametro", "omissao"}
    assert (purpose_rows["diff_quartil"] == 3).all()
    assert purpose_rows["subiu"].all()


def test_resumo_motivos_por_componente_aggregates_by_juiz_componente_motivo():
    motivos_df = pd.DataFrame([
        {"juiz": "j", "componente": "purpose", "tool_uid": "t0", "motivo": "omissao", "diff_quartil": 2, "subiu": True},
        {"juiz": "j", "componente": "purpose", "tool_uid": "t1", "motivo": "omissao", "diff_quartil": -2, "subiu": False},
        {"juiz": "j", "componente": "purpose", "tool_uid": "t2", "motivo": MOTIVO_FALLBACK, "diff_quartil": -2, "subiu": False},
    ])

    result = resumo_motivos_por_componente(motivos_df)

    omissao = result[result["motivo"] == "omissao"].iloc[0]
    assert omissao["ocorrencias"] == 2
    assert omissao["n_subiu"] == 1
    assert omissao["n_desceu"] == 1
    assert omissao["pct_subiu"] == 50.0


def test_custo_latencia_por_juiz_e_cenario_separates_by_scenario():
    records = [
        _record(tool_uid="t0", cenario="description_only", scores=_scores(), input_tokens=100, latency_ms=1000),
        _record(tool_uid="t0", cenario="with_source", scores=_scores(), input_tokens=300, latency_ms=1500),
    ]

    result = custo_latencia_por_juiz_e_cenario(records)

    desc = result[result["cenario"] == "description_only"].iloc[0]
    src = result[result["cenario"] == "with_source"].iloc[0]
    assert desc["media_input_tokens"] == 100
    assert src["media_input_tokens"] == 300


def _usage_record(juiz, *, status="ok", input_tokens=0, cache_read_input_tokens=0, output_tokens=0,
                   evaluated_at="2026-09-14T12:00:00+00:00"):
    """Fixture mínima para as funções de custo real (custo_real_usd/custo_real_por_juiz), que só
    leem judge.id, status, usage e evaluated_at -- não precisam do schema completo de _record()
    (scores/tool/repo). Default de evaluated_at é uma segunda-feira (2026-09-14) fora das
    janelas de peak da DeepSeek (ver _deepseek_em_horario_peak_utc()).
    """
    return {
        "judge": {"id": juiz},
        "status": status,
        "usage": {"input_tokens": input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "output_tokens": output_tokens},
        "evaluated_at": evaluated_at,
    }


def test_deepseek_em_horario_peak_utc_true_on_weekday_peak_window():
    assert _deepseek_em_horario_peak_utc("2026-09-14T02:00:00+00:00")  # segunda, 02h UTC (janela 01h-04h)


def test_deepseek_em_horario_peak_utc_false_outside_peak_windows():
    assert not _deepseek_em_horario_peak_utc("2026-09-14T12:00:00+00:00")  # segunda, meio-dia
    assert not _deepseek_em_horario_peak_utc("2026-09-14T05:00:00+00:00")  # segunda, entre as duas janelas de peak


def test_deepseek_em_horario_peak_utc_false_on_weekend_even_in_peak_window():
    assert not _deepseek_em_horario_peak_utc("2026-09-19T02:00:00+00:00")  # sábado, 02h UTC


def test_custo_real_usd_is_zero_for_free_tier_judges():
    record = _usage_record("gemini-3.5-flash-lite", input_tokens=100_000, output_tokens=50_000)
    assert custo_real_usd(record) == 0.0


def test_custo_real_usd_is_none_for_unmodeled_paid_judge():
    """Um juiz pago sem preço definido em DEEPSEEK_FLASH_PRICING_USD_POR_1M/
    JUDGES_SEM_CUSTO_DIRETO deve virar None, não $0.0 -- ver docstring de custo_real_usd()."""
    record = _usage_record("some-future-paid-judge", input_tokens=1000, output_tokens=500)
    assert custo_real_usd(record) is None


def test_custo_real_usd_computes_deepseek_off_peak_price():
    record = _usage_record(
        "deepseek-flash", input_tokens=1000, cache_read_input_tokens=200, output_tokens=500,
        evaluated_at="2026-09-14T12:00:00+00:00",  # segunda, off-peak
    )
    # cache_miss=800: 800*0.15 (input cache-miss) + 200*0.003 (input cache-hit) + 500*0.60 (output) = 420.6, por 1M tokens.
    assert custo_real_usd(record) == pytest.approx(420.6 / 1_000_000)


def test_custo_real_usd_computes_deepseek_peak_price_as_double_off_peak():
    record = _usage_record(
        "deepseek-flash", input_tokens=1000, cache_read_input_tokens=200, output_tokens=500,
        evaluated_at="2026-09-14T02:00:00+00:00",  # segunda, peak
    )
    assert custo_real_usd(record) == pytest.approx(2 * (420.6 / 1_000_000))


def test_custo_real_por_juiz_aggregates_and_flags_gratuito_e_nao_modelado():
    records = [
        _usage_record("deepseek-flash", input_tokens=1000, output_tokens=500, evaluated_at="2026-09-14T12:00:00+00:00"),
        _usage_record("deepseek-flash", input_tokens=1000, output_tokens=500, evaluated_at="2026-09-14T12:00:00+00:00"),
        _usage_record("gemini-3.5-flash-lite", input_tokens=999_999, output_tokens=999_999),
        _usage_record("some-future-paid-judge", input_tokens=1000, output_tokens=500),
        _usage_record("deepseek-flash", status="error", input_tokens=1000, output_tokens=500),  # ignorado (não "ok")
    ]

    result = custo_real_por_juiz(records).set_index("juiz")

    assert result.loc["deepseek-flash", "avaliacoes"] == 2
    assert result.loc["deepseek-flash", "custo_total_usd"] == pytest.approx(2 * 450 / 1_000_000)
    assert not result.loc["deepseek-flash", "gratuito"]
    assert result.loc["gemini-3.5-flash-lite", "custo_total_usd"] == 0.0
    assert result.loc["gemini-3.5-flash-lite", "gratuito"]
    assert result.loc["some-future-paid-judge", "custo_nao_modelado"] == 1


def test_delta_custo_com_codigo_computes_extra_cost_of_with_source():
    custo_df = pd.DataFrame([
        {"juiz": "j", "cenario": "description_only", "avaliacoes": 2, "media_input_tokens": 100.0, "media_output_tokens": 50.0, "media_latencia_ms": 1000.0, "media_custo_usd": 0.01},
        {"juiz": "j", "cenario": "with_source", "avaliacoes": 2, "media_input_tokens": 300.0, "media_output_tokens": 50.0, "media_latencia_ms": 1500.0, "media_custo_usd": 0.04},
    ])

    result = _delta_custo_com_codigo(custo_df)

    row = result.iloc[0]
    assert row["delta_input_tokens"] == 200.0
    assert row["delta_latencia_ms"] == 500.0
    assert row["custo_percentual_extra"] == 200.0
    assert row["delta_custo_usd"] == pytest.approx(0.03)


def _uniform_shift_records(desc_group_a, desc_group_b, src_group_a, src_group_b, juiz="j", prompt_version="v4", input_tokens_desc=100, input_tokens_src=400, latency_desc=1000.0, latency_src=1600.0):
    """5 tools com nota description_only = desc_group_a, 5 com desc_group_b; with_source
    correspondente em src_group_a/src_group_b -- desenhado para que o rank percentual dentro
    de cada cenário preserve a mesma forma (2 grupos de 5 empatados) nos dois cenários, então
    nenhuma tool migra de quartil (n_diverge=0), isolando o efeito de significância/custo do
    efeito de divergência em veredito_custo_beneficio().
    """
    records = []
    for i in range(5):
        records.append(_record(tool_uid=f"a{i}", cenario="description_only", juiz=juiz, prompt_version=prompt_version,
                                scores=_scores({"purpose": {"score": desc_group_a, "reasoning": "n/a"}}),
                                input_tokens=input_tokens_desc, latency_ms=latency_desc))
        records.append(_record(tool_uid=f"a{i}", cenario="with_source", juiz=juiz, prompt_version=prompt_version,
                                scores=_scores({"purpose": {"score": src_group_a, "reasoning": "n/a"}}),
                                input_tokens=input_tokens_src, latency_ms=latency_src))
        records.append(_record(tool_uid=f"b{i}", cenario="description_only", juiz=juiz, prompt_version=prompt_version,
                                scores=_scores({"purpose": {"score": desc_group_b, "reasoning": "n/a"}}),
                                input_tokens=input_tokens_desc, latency_ms=latency_desc))
        records.append(_record(tool_uid=f"b{i}", cenario="with_source", juiz=juiz, prompt_version=prompt_version,
                                scores=_scores({"purpose": {"score": src_group_b, "reasoning": "n/a"}}),
                                input_tokens=input_tokens_src, latency_ms=latency_src))
    return records


def test_veredito_custo_beneficio_flags_sim_when_significant_with_no_divergences_to_doubt():
    """with_source consistentemente 3 pontos ABAIXO (4->1, 5->2): diferença pareada uniforme e
    forte o bastante para dar Wilcoxon significativo, mas sem nenhuma migração de quartil (ver
    _uniform_shift_records()) -- pct_sem_motivo fica None, e a regra de vale_a_pena trata None
    como "não há divergência para desconfiar da justificativa", não como sinal negativo.
    """
    records = _uniform_shift_records(desc_group_a=4, desc_group_b=5, src_group_a=1, src_group_b=2)

    result = veredito_custo_beneficio(records)

    row = result[result["componente"] == "purpose"].iloc[0]
    assert row["direcao"] == "desce"
    assert row["significativo_bh_0.05"]
    assert row["pct_sem_motivo"] is None
    assert row["vale_a_pena"] == "sim"
    assert row["delta_input_tokens"] == pytest.approx(300.0)
    assert row["delta_latencia_ms"] == pytest.approx(600.0)


def _records_single_divergent_tool_biased_up(reasoning_do_divergente, juiz="j", prompt_version="v4"):
    """8 tools, componente "purpose": description_only com ranks exatos 1..8 (sem empates,
    quartis limpos [Q1,Q1,Q2,Q2,Q3,Q3,Q4,Q4] -- ver quartil_notas()); with_source preserva a
    ordem relativa de 7 delas (+10 cada, sem migração de quartil) e só t0 salta para o topo
    (Q1->Q4, diverge=True, subiu=True) -- isola exatamente 1 divergência, cujo motivo depende
    só de `reasoning_do_divergente`. Os 8 diffs pareados são todos positivos (+10 ou +99), o
    bastante (n_efetivo=8) para dar Wilcoxon significativo.
    """
    desc_scores = list(range(1, 9))
    src_scores = [100] + [d + 10 for d in desc_scores[1:]]
    records = []
    for i, (desc, src) in enumerate(zip(desc_scores, src_scores)):
        reasoning = reasoning_do_divergente if i == 0 else "fine"
        records.append(_record(tool_uid=f"t{i}", cenario="description_only", juiz=juiz, prompt_version=prompt_version,
                                scores=_scores({"purpose": {"score": desc, "reasoning": "n/a"}})))
        records.append(_record(tool_uid=f"t{i}", cenario="with_source", juiz=juiz, prompt_version=prompt_version,
                                scores=_scores({"purpose": {"score": src, "reasoning": reasoning}})))
    return records


def test_veredito_custo_beneficio_flags_nao_when_significant_but_unjustified():
    """Direção sozinha não decide mais o veredito (o texto ativo do prompt não restringe para
    qual lado a nota pode mudar, ver docstring de veredito_custo_beneficio()) -- quem decide é
    a justificativa. Único tool divergente com reasoning genérica (sem motivo casado) ->
    pct_sem_motivo=100% -> "não", mesmo com efeito significativo."""
    records = _records_single_divergent_tool_biased_up("Looks fine overall.")

    result = veredito_custo_beneficio(records)

    row = result[result["componente"] == "purpose"].iloc[0]
    assert row["direcao"] == "sobe"
    assert row["significativo_bh_0.05"]
    assert row["pct_sem_motivo"] == 100.0
    assert row["vale_a_pena"] == "não"


def test_veredito_custo_beneficio_flags_sim_when_significant_and_justified():
    """Mesmo desenho do teste acima, mas a reasoning do único tool divergente nomeia um
    problema concreto -> pct_sem_motivo=0% -> "sim", apesar de ser a mesma direção "sobe" do
    teste anterior -- prova que a direção sozinha não determina o veredito."""
    records = _records_single_divergent_tool_biased_up("The parameter type is missing.")

    result = veredito_custo_beneficio(records)

    row = result[result["componente"] == "purpose"].iloc[0]
    assert row["direcao"] == "sobe"
    assert row["pct_sem_motivo"] == 0.0
    assert row["vale_a_pena"] == "sim"


def test_veredito_custo_beneficio_excludes_older_prompt_versions():
    """Registros de uma versão de prompt anterior (v3) não devem aparecer no veredito, mesmo
    vindo de um juiz que só existe nessa versão -- ver registros_versao_ativa()."""
    v4_records = _uniform_shift_records(desc_group_a=1, desc_group_b=2, src_group_a=4, src_group_b=5)
    v3_records = _uniform_shift_records(
        desc_group_a=1, desc_group_b=2, src_group_a=5, src_group_b=5,
        juiz="old_judge", prompt_version="v3",
    )

    result = veredito_custo_beneficio(v4_records + v3_records)

    assert "old_judge" not in set(result["juiz"])
