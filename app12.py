import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, replace
from typing import Optional
import numpy as np
import math

# -------------------------------------------------------------------
# LÓGICA DE NEGÓCIO
# -------------------------------------------------------------------

def annual_to_monthly(rate_annual_pct: float) -> float:
    return (1 + rate_annual_pct / 100.0) ** (1 / 12.0) - 1

def fmt_brl(value: float) -> str:
    neg = value < 0
    value = abs(value)
    s = f"{value:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"-R$ {s}" if neg else f"R$ {s}"

@dataclass
class Inputs:
    initial_cash: float
    property_price_today: float
    property_appreciation_annual: float
    investment_return_annual_net: float
    horizon_years: int
    cash_purchase_year: float
    cash_purchase_extra_cost_pct: float
    consortium_credit_today: float
    admin_fee_pct: float
    reserve_fund_pct: float
    insurance_monthly_pct_of_credit: float
    term_months: int
    contemplation_year: float
    pre_contemplation_payment_fraction: float
    bid_pct: float
    embedded_bid_pct_of_bid: float
    bid_base: str
    adjustment_pre_annual: float
    adjustment_post_annual: float
    reduce_installment_after_bid: bool
    monthly_rent_saved_after_purchase: float
    rent_inflation_annual: float
    vacancy_months_initial: int
    monthly_property_cost_after_purchase: float
    monthly_precision_rounding: bool
    monthly_savings: float
    savings_inflation_annual: float
    ir_aliquota_pct: float
    investment_return_annual_gross: float
    no_cash_purchase: bool  # Se True, estratégia "à vista" nunca compra — só deixa rendendo
    # ====== FLAGS DE INCLUSÃO NA COMPARAÇÃO ======
    cash_enabled: bool
    cons_enabled: bool
    # ====== FINANCIAMENTO ======
    financing_enabled: bool
    financing_start_year: float
    financing_down_payment_pct: float
    financing_interest_annual: float
    financing_term_months: int
    financing_system: str  # "SAC" ou "Price"
    financing_extra_cost_pct: float
    financing_insurance_monthly_pct_of_balance: float
    financing_balance_adjustment_annual: float
    prepayment_amount: float
    prepayment_period_months: int
    prepayment_adjustment_annual: float
    prepayment_mode: str  # "reduce_term" ou "reduce_installment"

def missing_required_fields(inp: Inputs) -> list[str]:
    """Retorna lista de campos obrigatórios ainda não preenchidos (valor 0)."""
    missing = []
    if inp.initial_cash <= 0:
        missing.append("Caixa inicial")
    if inp.property_price_today <= 0:
        missing.append("Preço do imóvel hoje")
    # Carta de crédito só é obrigatória se o consórcio está incluído na comparação
    if inp.cons_enabled and inp.consortium_credit_today <= 0:
        missing.append("Carta de crédito do consórcio")
    return missing

def validate_inputs(inp: Inputs) -> list[str]:
    """Valida inputs e retorna lista de erros (apenas inconsistências, não campos zerados)."""
    errors = []
    if inp.horizon_years <= 0:
        errors.append("Horizonte deve ser maior que zero.")
    if not inp.no_cash_purchase:
        if inp.cash_purchase_year > inp.horizon_years:
            errors.append("Ano da compra à vista não pode exceder o horizonte.")
        if inp.cash_purchase_year <= 0:
            errors.append("Ano da compra à vista deve ser maior que zero.")
    if inp.contemplation_year > inp.horizon_years:
        errors.append("Ano de contemplação não pode exceder o horizonte.")
    if inp.contemplation_year <= 0:
        errors.append("Ano de contemplação deve ser maior que zero.")
    if inp.term_months <= 0:
        errors.append("Prazo do consórcio deve ser maior que zero.")
    if inp.bid_pct < 0 or inp.bid_pct > 100:
        errors.append("Percentual do lance deve estar entre 0% e 100%.")
    if inp.embedded_bid_pct_of_bid < 0 or inp.embedded_bid_pct_of_bid > 100:
        errors.append("Percentual do lance embutido deve estar entre 0% e 100%.")
    if inp.pre_contemplation_payment_fraction <= 0 or inp.pre_contemplation_payment_fraction > 1:
        errors.append("Fração da parcela antes da contemplação deve estar entre 0 e 1.")
    if inp.admin_fee_pct < 0:
        errors.append("Taxa administrativa não pode ser negativa.")
    if inp.monthly_savings < 0:
        errors.append("Aporte mensal não pode ser negativo.")
    if inp.vacancy_months_initial < 0:
        errors.append("Vacância inicial não pode ser negativa.")
    # Validações do financiamento (só quando habilitado)
    if inp.financing_enabled:
        if inp.financing_start_year <= 0:
            errors.append("Ano de contratação do financiamento deve ser maior que zero.")
        if inp.financing_start_year > inp.horizon_years:
            errors.append("Ano de contratação do financiamento não pode exceder o horizonte.")
        if inp.financing_down_payment_pct < 0 or inp.financing_down_payment_pct > 100:
            errors.append("Entrada do financiamento deve estar entre 0% e 100%.")
        if inp.financing_interest_annual < 0:
            errors.append("Taxa de juros do financiamento não pode ser negativa.")
        if inp.financing_term_months <= 0:
            errors.append("Prazo do financiamento deve ser maior que zero.")
        if inp.financing_system not in ("SAC", "Price"):
            errors.append("Sistema de amortização inválido (use SAC ou Price).")
        if inp.financing_extra_cost_pct < 0:
            errors.append("Custos extras de aquisição não podem ser negativos.")
        if inp.financing_insurance_monthly_pct_of_balance < 0:
            errors.append("Seguro do financiamento não pode ser negativo.")
        if inp.prepayment_amount < 0:
            errors.append("Valor da amortização antecipada não pode ser negativo.")
        if inp.prepayment_period_months < 0:
            errors.append("Periodicidade da amortização antecipada não pode ser negativa.")
        if inp.prepayment_mode not in ("reduce_term", "reduce_installment"):
            errors.append("Modo de amortização antecipada inválido.")
    if inp.ir_aliquota_pct < 0 or inp.ir_aliquota_pct > 100:
        errors.append("Alíquota de IR deve estar entre 0% e 100%.")
    return errors

def round_if_needed(x: float, enabled: bool) -> float:
    return round(x, 2) if enabled else x

def property_value_at_month(initial_price: float, annual_app: float, month: int) -> float:
    return initial_price * ((1 + annual_app / 100.0) ** (month / 12.0))

def rental_value_at_month(base_rent_today: float, rent_inflation_annual: float, month: int) -> float:
    if base_rent_today <= 0:
        return 0.0
    return base_rent_today * ((1 + rent_inflation_annual / 100.0) ** (month / 12.0))

def savings_value_at_month(base_savings: float, savings_inflation_annual: float, month: int) -> float:
    if base_savings <= 0:
        return 0.0
    return base_savings * ((1 + savings_inflation_annual / 100.0) ** (month / 12.0))

def simulate_cash_later(inp: Inputs) -> dict:
    r_m = annual_to_monthly(inp.investment_return_annual_net)
    # IR informacional: estimativa baseada na taxa bruta vs líquida
    r_m_gross = annual_to_monthly(inp.investment_return_annual_gross)
    total_months = int(round(inp.horizon_years * 12))
    purchase_month = int(round(inp.cash_purchase_year * 12))

    balance = inp.initial_cash
    purchased = False
    purchase_executed_month: Optional[int] = None
    records = []
    cumulative_purchase_outflow = 0.0
    cumulative_property_cost = 0.0
    cumulative_rent_saved = 0.0
    cumulative_savings = 0.0
    cumulative_ir_paid = 0.0
    out_of_pocket_total = 0.0
    went_negative = False

    for m in range(total_months + 1):
        property_value = property_value_at_month(inp.property_price_today, inp.property_appreciation_annual, m)
        purchase_outflow = 0.0

        if m > 0:
            # Aporte mensal corrigido pela inflação
            monthly_contribution = savings_value_at_month(inp.monthly_savings, inp.savings_inflation_annual, m)
            balance += monthly_contribution
            cumulative_savings += monthly_contribution

            investment_return = balance * r_m
            gross_return = balance * r_m_gross
            ir_paid = gross_return * (inp.ir_aliquota_pct / 100.0)
            balance += investment_return
            cumulative_ir_paid += ir_paid

            if (not purchased) and (not inp.no_cash_purchase) and (m == purchase_month):
                purchase_outflow = property_value * (1 + inp.cash_purchase_extra_cost_pct / 100.0)
                balance -= purchase_outflow
                cumulative_purchase_outflow += purchase_outflow
                out_of_pocket_total += purchase_outflow
                purchased = True
                purchase_executed_month = m

            if purchased:
                # Vacância: aluguel só começa a contar após N meses da compra
                months_since_purchase = m - (purchase_executed_month or m)
                if months_since_purchase >= inp.vacancy_months_initial:
                    rent_saved = rental_value_at_month(inp.monthly_rent_saved_after_purchase, inp.rent_inflation_annual, m)
                else:
                    rent_saved = 0.0
                property_cost = inp.monthly_property_cost_after_purchase
                balance += rent_saved - property_cost
                cumulative_rent_saved += rent_saved
                cumulative_property_cost += property_cost

            if balance < 0:
                went_negative = True

            balance = round_if_needed(balance, inp.monthly_precision_rounding)

        house_component = property_value if purchased else 0.0
        # Na estratégia à vista, não há parcela — o desembolso ocorre só no mês da compra
        records.append({
            "month": m,
            "year": m / 12.0,
            "financial_wealth": balance,
            "real_estate_wealth": house_component,
            "total_wealth": balance + house_component,
            "monthly_outflow": purchase_outflow,  # só no mês da compra, zero nos demais
        })

    return {
        "records": records,
        "final_financial_wealth": records[-1]["financial_wealth"],
        "final_real_estate_wealth": records[-1]["real_estate_wealth"],
        "final_total_wealth": records[-1]["total_wealth"],
        "purchase_month": purchase_month,
        "cumulative_purchase_outflow": cumulative_purchase_outflow,
        "cumulative_savings": cumulative_savings,
        "cumulative_ir_paid": cumulative_ir_paid,
        "out_of_pocket_total": out_of_pocket_total,
        "went_negative": went_negative,
    }

def simulate_consortium(inp: Inputs) -> dict:
    r_m = annual_to_monthly(inp.investment_return_annual_net)
    r_m_gross = annual_to_monthly(inp.investment_return_annual_gross)
    total_months = int(round(inp.horizon_years * 12))
    contemplation_month = int(round(inp.contemplation_year * 12))

    # Garante que contemplation_month >= 1 para evitar divisão por zero e None em credit_reference
    contemplation_month = max(contemplation_month, 1)

    balance = inp.initial_cash
    purchased = False
    records = []

    total_plan_today = inp.consortium_credit_today * (
        1 + inp.admin_fee_pct / 100.0 + inp.reserve_fund_pct / 100.0
    )
    pre_adj_m = annual_to_monthly(inp.adjustment_pre_annual)
    post_adj_m = annual_to_monthly(inp.adjustment_post_annual)

    cumulative_installments = 0.0
    cumulative_insurance = 0.0
    cumulative_bid_own_cash = 0.0
    cumulative_bid_embedded = 0.0
    cumulative_property_topup = 0.0
    cumulative_savings = 0.0
    cumulative_ir_paid = 0.0
    out_of_pocket_total = 0.0
    went_negative = False

    post_balance_debt: Optional[float] = None
    credit_reference_at_contemplation: Optional[float] = None
    debt_at_contemplation: Optional[float] = None
    effective_credit_at_contemplation: Optional[float] = None
    current_installment: float = 0.0

    # Calcula o plano reajustado no mês da contemplação (reajuste anual em degrau)
    years_at_contemplation = contemplation_month // 12
    plan_at_contemplation = total_plan_today * ((1 + inp.adjustment_pre_annual / 100.0) ** years_at_contemplation)

    # Número de parcelas já pagas até a contemplação (exclusive)
    installments_paid_pre = 0

    for m in range(total_months + 1):
        # Referência de crédito no mês m
        if m <= contemplation_month:
            current_credit_reference = inp.consortium_credit_today * ((1 + pre_adj_m) ** m)
        else:
            months_after = m - contemplation_month
            current_credit_reference = credit_reference_at_contemplation * ((1 + post_adj_m) ** months_after)

        property_value = property_value_at_month(inp.property_price_today, inp.property_appreciation_annual, m)

        if m > 0:
            # Aporte mensal corrigido pela inflação
            monthly_contribution = savings_value_at_month(inp.monthly_savings, inp.savings_inflation_annual, m)
            balance += monthly_contribution
            cumulative_savings += monthly_contribution

            gross_return = balance * r_m_gross
            ir_paid = gross_return * (inp.ir_aliquota_pct / 100.0)
            balance += balance * r_m
            cumulative_ir_paid += ir_paid

            current_installment = 0.0  # parcela do mês corrente

            if m <= contemplation_month:
                # Parcela proporcional pré-contemplação
                # Reajuste anual em degrau (mesma parcela por 12 meses, sobe no aniversário)
                years_elapsed = (m - 1) // 12
                current_plan_value = total_plan_today * ((1 + inp.adjustment_pre_annual / 100.0) ** years_elapsed)
                full_installment = current_plan_value / inp.term_months
                installment = full_installment * inp.pre_contemplation_payment_fraction
                insurance = current_credit_reference * (inp.insurance_monthly_pct_of_credit / 100.0)

                balance -= (installment + insurance)
                cumulative_installments += installment
                cumulative_insurance += insurance
                out_of_pocket_total += installment + insurance
                installments_paid_pre += 1
                current_installment = installment + insurance

                # Contemplação acontece no final deste mês
                if m == contemplation_month:
                    # Cálculo do lance — sobre o plano total reajustado (carta + taxa adm + fundo reserva)
                    if inp.bid_base == "original_credit":
                        total_bid = total_plan_today * inp.bid_pct / 100.0
                    else:
                        total_bid = plan_at_contemplation * inp.bid_pct / 100.0

                    embedded_bid = total_bid * (inp.embedded_bid_pct_of_bid / 100.0)
                    own_cash_bid = total_bid - embedded_bid

                    balance -= own_cash_bid
                    cumulative_bid_own_cash += own_cash_bid
                    cumulative_bid_embedded += embedded_bid
                    out_of_pocket_total += own_cash_bid
                    purchased = True

                    effective_credit_received = max(current_credit_reference - embedded_bid, 0.0)
                    effective_credit_at_contemplation = effective_credit_received
                    credit_reference_at_contemplation = current_credit_reference

                    # Complemento se imóvel vale mais que a carta efetiva
                    if effective_credit_received < property_value:
                        property_topup = property_value - effective_credit_received
                        balance -= property_topup
                        cumulative_property_topup += property_topup
                        out_of_pocket_total += property_topup

                    # Saldo devedor remanescente do plano
                    # O saldo é o total do plano reajustado menos o que já foi pago
                    # (considerando ou não a abatimento do lance)
                    if inp.reduce_installment_after_bid:
                        post_balance_debt = max(plan_at_contemplation - cumulative_installments - total_bid, 0.0)
                    else:
                        post_balance_debt = max(plan_at_contemplation - cumulative_installments, 0.0)

                    debt_at_contemplation = post_balance_debt

            else:
                # Pós-contemplação: parcela fixa + reajuste anual em degrau
                # Parcela base = saldo devedor na contemplação ÷ meses restantes no plano
                # Reajusta 1x por ano pelo índice pós (INCC/IPCA)
                # Paga até o mês final do plano (term_months), independente do saldo contábil
                months_after_contemplation = m - contemplation_month
                months_elapsed_in_plan = installments_paid_pre + months_after_contemplation

                if months_elapsed_in_plan > inp.term_months:
                    # Plano encerrado
                    installment = 0.0
                    insurance = 0.0
                    post_balance_debt = 0.0
                else:
                    # Parcela base definida UMA VEZ na contemplação
                    total_remaining_at_contemplation = inp.term_months - installments_paid_pre
                    base_installment = debt_at_contemplation / max(total_remaining_at_contemplation, 1)

                    # Reajuste anual em degrau (meses 1-12 pós = base, 13-24 = +5%, etc.)
                    years_post = (months_after_contemplation - 1) // 12
                    installment = base_installment * ((1 + inp.adjustment_post_annual / 100.0) ** years_post)

                    insurance = current_credit_reference * (inp.insurance_monthly_pct_of_credit / 100.0)

                    # Saldo devedor contábil (para exibição — decresce linearmente pela parcela base)
                    parcelas_pos_pagas = months_after_contemplation
                    post_balance_debt = max(debt_at_contemplation - (base_installment * parcelas_pos_pagas), 0.0)

                balance -= (installment + insurance)
                cumulative_installments += installment
                cumulative_insurance += insurance
                out_of_pocket_total += installment + insurance
                current_installment = installment + insurance

            if purchased:
                # Vacância: aluguel só começa a contar após N meses da contemplação
                months_since_purchase = m - contemplation_month
                if months_since_purchase >= inp.vacancy_months_initial:
                    rent_saved = rental_value_at_month(
                        inp.monthly_rent_saved_after_purchase, inp.rent_inflation_annual, m
                    )
                else:
                    rent_saved = 0.0
                balance += rent_saved - inp.monthly_property_cost_after_purchase

            if balance < 0:
                went_negative = True

            balance = round_if_needed(balance, inp.monthly_precision_rounding)

        house_component = property_value if purchased else 0.0
        remaining = post_balance_debt if post_balance_debt is not None else 0.0
        records.append({
            "month": m,
            "year": m / 12.0,
            "financial_wealth": balance,
            "real_estate_wealth": house_component,
            "remaining_debt": remaining,
            "total_wealth": balance + house_component,
            "monthly_outflow": current_installment if m > 0 else 0.0,
        })

    return {
        "records": records,
        "final_financial_wealth": records[-1]["financial_wealth"],
        "final_real_estate_wealth": records[-1]["real_estate_wealth"],
        "final_total_wealth": records[-1]["total_wealth"],
        "contemplation_month": contemplation_month,
        "debt_at_contemplation": debt_at_contemplation or 0.0,
        "effective_credit_at_contemplation": effective_credit_at_contemplation or 0.0,
        "final_remaining_debt": records[-1]["remaining_debt"],
        "final_cumulative_installments": cumulative_installments,
        "final_cumulative_insurance": cumulative_insurance,
        "final_cumulative_bid_own_cash": cumulative_bid_own_cash,
        "final_cumulative_bid_embedded": cumulative_bid_embedded,
        "final_cumulative_property_topup": cumulative_property_topup,
        "cumulative_savings": cumulative_savings,
        "cumulative_ir_paid": cumulative_ir_paid,
        "out_of_pocket_total": out_of_pocket_total,
        "went_negative": went_negative,
    }


# -------------------------------------------------------------------
# ANÁLISE DE SENSIBILIDADE
# -------------------------------------------------------------------

SENSITIVITY_PARAMS = {
    "investment_return_annual_net": "Rentabilidade líquida a.a. (%)",
    "property_appreciation_annual": "Valorização do imóvel a.a. (%)",
    "bid_pct": "Lance total (%)",
    "contemplation_year": "Ano de contemplação",
    "cash_purchase_year": "Ano da compra à vista",
    "admin_fee_pct": "Taxa administrativa (%)",
    "adjustment_pre_annual": "Reajuste pré-contemplação INCC (%)",
    "adjustment_post_annual": "Reajuste pós-contemplação IPCA (%)",
}

def sensitivity_sweep(base_inp: Inputs, param: str, values: list[float]) -> list[dict]:
    """Varia um parâmetro e retorna o patrimônio final de cada estratégia."""
    results = []
    for v in values:
        inp_v = replace(base_inp, **{param: v})
        errors = validate_inputs(inp_v)
        if errors:
            continue
        try:
            cr = simulate_cash_later(inp_v)
            co = simulate_consortium(inp_v)
            results.append({
                "value": v,
                "cash_total": cr["final_total_wealth"],
                "cons_total": co["final_total_wealth"],
                "diff": co["final_total_wealth"] - cr["final_total_wealth"],
            })
        except Exception:
            continue
    return results

def simulate_financing(inp: Inputs) -> dict:
    """
    Simulação da estratégia de financiamento imobiliário.

    Mecânica:
    - No mês da contratação, paga-se entrada (% do preço do imóvel no mês) + custos extras.
    - O saldo restante é financiado. Calculam-se parcelas conforme o sistema (SAC ou Price).
    - Mês a mês, paga-se parcela = amortização + juros + seguro.
    - O saldo devedor é corrigido anualmente (TR) e amortizado conforme o sistema.
    - Opcionalmente há amortizações extraordinárias periódicas, com duas políticas:
      * reduce_term: mantém parcela, recalcula prazo (paga antes)
      * reduce_installment: mantém prazo, recalcula parcela (parcela fica menor)
    - Se o caixa não comporta uma amortização extraordinária, ela é PULADA (não ocorre).

    Matemática SAC (Sistema de Amortização Constante):
    - Amortização fixa = saldo / prazo_restante
    - Juros = saldo * taxa_mensal
    - Parcela mês m = amortização + juros(m)  (decrescente)

    Matemática Price (Tabela Price):
    - Parcela = saldo * [i * (1+i)^n] / [(1+i)^n - 1], onde n é prazo restante e i é a taxa mensal
    - Juros = saldo * i
    - Amortização = parcela - juros
    """
    total_months = inp.horizon_years * 12
    r_m = annual_to_monthly(inp.investment_return_annual_net)
    r_m_gross = annual_to_monthly(inp.investment_return_annual_gross)
    financing_month = int(round(inp.financing_start_year * 12))

    # Se financiamento desabilitado, retorna resultado vazio/equivalente à estratégia "só rende"
    if not inp.financing_enabled:
        return {
            "records": [],
            "final_total_wealth": 0.0,
            "final_financial_wealth": 0.0,
            "final_real_estate_wealth": 0.0,
            "out_of_pocket_total": 0.0,
            "final_remaining_debt": 0.0,
            "final_cumulative_installments": 0.0,
            "final_cumulative_insurance": 0.0,
            "final_cumulative_interest": 0.0,
            "final_cumulative_amortization": 0.0,
            "final_cumulative_prepayments": 0.0,
            "prepayments_skipped": 0,
            "down_payment": 0.0,
            "acquisition_extra": 0.0,
            "financed_amount": 0.0,
            "went_negative": False,
            "enabled": False,
        }

    # Estado do financiamento
    balance = inp.initial_cash
    purchased = False
    debt = 0.0  # saldo devedor
    remaining_term = 0  # prazo restante em meses
    price_installment = 0.0  # parcela Price vigente
    i_m = annual_to_monthly(inp.financing_interest_annual)  # juros mensais do financ.

    # Acumuladores
    records = []
    cumulative_installments = 0.0
    cumulative_insurance = 0.0
    cumulative_interest = 0.0
    cumulative_amortization = 0.0
    cumulative_savings = 0.0
    cumulative_ir_paid = 0.0
    cumulative_prepayments = 0.0
    prepayments_skipped = 0
    out_of_pocket_total = 0.0
    down_payment = 0.0
    acquisition_extra = 0.0
    financed_amount = 0.0
    went_negative = False

    # Função auxiliar: recalcula a parcela Price a partir de (saldo, prazo restante, taxa)
    def price_payment(saldo: float, n: int, i: float) -> float:
        if n <= 0 or saldo <= 0:
            return 0.0
        if i <= 0:
            return saldo / n
        # Fórmula padrão Price
        factor = (1 + i) ** n
        return saldo * (i * factor) / (factor - 1)

    for m in range(total_months + 1):
        property_value = property_value_at_month(
            inp.property_price_today, inp.property_appreciation_annual, m
        )
        current_monthly_outflow = 0.0

        if m > 0:
            # 1. Aporte mensal corrigido
            monthly_contribution = savings_value_at_month(
                inp.monthly_savings, inp.savings_inflation_annual, m
            )
            balance += monthly_contribution
            cumulative_savings += monthly_contribution

            # 2. Rendimento do caixa
            investment_return = balance * r_m
            gross_return = balance * r_m_gross
            ir_paid = gross_return * (inp.ir_aliquota_pct / 100.0)
            balance += investment_return
            cumulative_ir_paid += ir_paid

            # 3. Contratação do financiamento (no mês marcado)
            if (not purchased) and (m == financing_month):
                down_payment = property_value * (inp.financing_down_payment_pct / 100.0)
                acquisition_extra = property_value * (inp.financing_extra_cost_pct / 100.0)
                total_upfront = down_payment + acquisition_extra
                balance -= total_upfront
                out_of_pocket_total += total_upfront
                debt = property_value - down_payment
                financed_amount = debt
                remaining_term = inp.financing_term_months
                if inp.financing_system == "Price":
                    price_installment = price_payment(debt, remaining_term, i_m)
                purchased = True

            # 4. Pagamento da parcela mensal (só a partir do mês seguinte à contratação)
            if purchased and debt > 0 and remaining_term > 0 and m > financing_month:
                # 4a. Correção anual do saldo devedor (TR/IPCA) — em degrau anual, no aniversário
                months_since_financing = m - financing_month
                if months_since_financing > 0 and months_since_financing % 12 == 0:
                    debt *= (1 + inp.financing_balance_adjustment_annual / 100.0)
                    # Price recalcula parcela após correção (saldo mudou)
                    if inp.financing_system == "Price":
                        price_installment = price_payment(debt, remaining_term, i_m)

                # 4b. Calcula juros do mês sobre o saldo atualizado
                interest_m = debt * i_m

                # 4c. Calcula amortização conforme sistema
                if inp.financing_system == "SAC":
                    amort = debt / remaining_term  # amortização constante
                    installment_principal = amort + interest_m
                else:  # Price
                    installment_principal = price_installment
                    amort = installment_principal - interest_m
                    if amort < 0:
                        amort = 0.0  # guarda contra edge cases

                # 4d. Seguro sobre saldo devedor ANTES do pagamento deste mês
                insurance_m = debt * (inp.financing_insurance_monthly_pct_of_balance / 100.0)

                # 4e. Total de saída do bolso no mês
                installment_total = installment_principal + insurance_m
                balance -= installment_total
                out_of_pocket_total += installment_total
                current_monthly_outflow = installment_total
                cumulative_installments += installment_principal
                cumulative_insurance += insurance_m
                cumulative_interest += interest_m
                cumulative_amortization += amort

                # 4f. Atualiza saldo devedor e prazo
                debt -= amort
                remaining_term -= 1
                if debt < 0.01:  # quitou
                    debt = 0.0
                    remaining_term = 0

                # 4g. Amortização extraordinária (periódica)
                if (inp.prepayment_amount > 0
                        and inp.prepayment_period_months > 0
                        and months_since_financing > 0
                        and months_since_financing % inp.prepayment_period_months == 0
                        and debt > 0
                        and remaining_term > 0):
                    # Valor da amortização corrigido anualmente
                    years_since_financing = months_since_financing / 12.0
                    adj_factor = (1 + inp.prepayment_adjustment_annual / 100.0) ** years_since_financing
                    prepay_amount = inp.prepayment_amount * adj_factor

                    # Não pode amortizar mais do que o saldo
                    prepay_amount = min(prepay_amount, debt)

                    # Se caixa não comporta: PULA (política (a))
                    if balance >= prepay_amount:
                        balance -= prepay_amount
                        debt -= prepay_amount
                        out_of_pocket_total += prepay_amount
                        cumulative_prepayments += prepay_amount
                        cumulative_amortization += prepay_amount

                        if debt < 0.01:
                            debt = 0.0
                            remaining_term = 0
                        else:
                            if inp.prepayment_mode == "reduce_term":
                                # Mantém parcela, recalcula prazo
                                if inp.financing_system == "SAC":
                                    # SAC: amortização constante = debt_original_no_início_do_contrato / n_orig
                                    # Mas mudou o saldo, então recalculamos o prazo necessário
                                    # mantendo a MESMA amortização mensal que estava vigente.
                                    # amortização mensal atual foi = debt_anterior / remaining_term_anterior
                                    # Vamos recalcular: novo prazo = ceil(debt_novo / amort_atual)
                                    if amort > 0:
                                        new_term = int(math.ceil(debt / amort))
                                        remaining_term = max(1, new_term)
                                else:
                                    # Price: mantém parcela fixa, recalcula prazo
                                    # n = -log(1 - debt*i/PMT) / log(1+i)
                                    if price_installment > 0 and i_m > 0:
                                        ratio = debt * i_m / price_installment
                                        if ratio < 1:
                                            new_term = int(math.ceil(
                                                -math.log(1 - ratio) / math.log(1 + i_m)
                                            ))
                                            remaining_term = max(1, new_term)
                            else:  # reduce_installment
                                # Mantém prazo, recalcula parcela
                                if inp.financing_system == "Price":
                                    price_installment = price_payment(debt, remaining_term, i_m)
                                # SAC: nova amortização = debt / remaining_term (acontece naturalmente no próximo mês)

            if balance < 0:
                went_negative = True

            # 5. Após posse: aluguel/custo do imóvel (mesma lógica das outras estratégias)
            if purchased:
                months_since_purchase = m - financing_month
                if months_since_purchase >= inp.vacancy_months_initial:
                    rent_saved = rental_value_at_month(
                        inp.monthly_rent_saved_after_purchase, inp.rent_inflation_annual, m
                    )
                else:
                    rent_saved = 0.0
                balance += rent_saved - inp.monthly_property_cost_after_purchase

            balance = round_if_needed(balance, inp.monthly_precision_rounding)

        house_component = property_value if purchased else 0.0
        records.append({
            "month": m,
            "year": m / 12.0,
            "financial_wealth": balance,
            "real_estate_wealth": house_component,
            "remaining_debt": debt,
            "total_wealth": balance + house_component - debt,  # considera dívida como passivo
            "monthly_outflow": current_monthly_outflow,
        })

    return {
        "records": records,
        "final_total_wealth": balance + (property_value if purchased else 0.0) - debt,
        "final_financial_wealth": balance,
        "final_real_estate_wealth": property_value if purchased else 0.0,
        "out_of_pocket_total": out_of_pocket_total,
        "final_remaining_debt": debt,
        "final_cumulative_installments": cumulative_installments,
        "final_cumulative_insurance": cumulative_insurance,
        "final_cumulative_interest": cumulative_interest,
        "final_cumulative_amortization": cumulative_amortization,
        "final_cumulative_prepayments": cumulative_prepayments,
        "prepayments_skipped": prepayments_skipped,
        "cumulative_savings": cumulative_savings,
        "cumulative_ir_paid": cumulative_ir_paid,
        "down_payment": down_payment,
        "acquisition_extra": acquisition_extra,
        "financed_amount": financed_amount,
        "went_negative": went_negative,
        "enabled": True,
    }


def find_crossover(sweep_results: list[dict]) -> Optional[float]:
    """
    Encontra o valor do parâmetro onde consórcio e à vista se cruzam
    (onde a diferença muda de sinal), por interpolação linear.
    """
    for i in range(len(sweep_results) - 1):
        d0 = sweep_results[i]["diff"]
        d1 = sweep_results[i + 1]["diff"]
        if d0 * d1 <= 0 and d0 != d1:
            v0 = sweep_results[i]["value"]
            v1 = sweep_results[i + 1]["value"]
            # Interpolação linear: onde diff = 0
            crossover = v0 + (-d0) * (v1 - v0) / (d1 - d0)
            return crossover
    return None


# -------------------------------------------------------------------
# INTERFACE WEB (STREAMLIT)
# -------------------------------------------------------------------

st.set_page_config(page_title="Simulador: Consórcio vs À Vista", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.1rem; }
    .winner-box {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .winner-consortium { background-color: #fff3cd; border-left: 5px solid #f0a500; }
    .winner-cash { background-color: #d1ecf1; border-left: 5px solid #0077a8; }
    .winner-financing { background-color: #d4edda; border-left: 5px solid #28a745; }
</style>
""", unsafe_allow_html=True)

st.title("🏡 Simulador Financeiro: Consórcio vs Compra à Vista vs Financiamento")

st.markdown("""
Este simulador compara, mês a mês, três formas de obter um imóvel:

- **💵 Compra à Vista** — você mantém o dinheiro investido rendendo e saca no ano escolhido para comprar o imóvel à vista.
- **🏦 Consórcio** — você paga parcelas mensais, é contemplado em um determinado ano (por lance), recebe a carta de crédito e adquire o imóvel. O restante do dinheiro segue rendendo no caixa.
- **🏛️ Financiamento** — você paga entrada (tipicamente 20%), assume uma dívida com o banco e toma posse imediata. Paga parcelas mensais até quitar, e opcionalmente faz amortizações antecipadas.

Ao final do horizonte, o simulador compara o **patrimônio total** (caixa investido + valor do imóvel − dívidas residuais) das estratégias. Ajuste os parâmetros na barra lateral à esquerda — cada campo tem uma explicação 🛈 ao passar o mouse.

> 💡 **Dica:** comece preenchendo os três campos obrigatórios (caixa inicial, preço do imóvel e carta de crédito) em **Parâmetros → Gerais** e **Estratégia: Consórcio**.
""")

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Parâmetros da Simulação")

with st.sidebar.expander("📌 Gerais", expanded=True):
    initial_cash = st.number_input(
        "Caixa inicial (R$)", value=0.0, step=50_000.0, min_value=0.0, format="%.2f",
        help="Quanto dinheiro você tem disponível hoje para investir. É o ponto de partida do caixa nas duas estratégias."
    )
    property_price_today = st.number_input(
        "Preço do imóvel hoje (R$)", value=0.0, step=50_000.0, min_value=0.0, format="%.2f",
        help="Valor de mercado do imóvel desejado, em reais de hoje. Será corrigido pela valorização anual ao longo do tempo."
    )
    property_appreciation_annual = st.number_input(
        "Valorização do imóvel a.a. (%)", value=5.0, step=0.1, format="%.2f",
        help="Quanto o imóvel se valoriza por ano. Referência histórica brasileira: FipeZap costuma ficar entre 3% e 7% a.a., dependendo da cidade e do período."
    )
    investment_return_annual_net = st.number_input(
        "Rentabilidade líquida a.a. (%)", value=8.5, step=0.1, format="%.2f",
        help="Taxa JÁ DESCONTADA de IR e taxas. Use a rentabilidade real que você espera obter no caixa (ex: LCI/LCA líquida, CDB líquido, Tesouro líquido). É aqui que sai o custo de oportunidade do dinheiro."
    )
    horizon_years = st.number_input(
        "Horizonte final (anos)", value=15, min_value=1, max_value=40, step=1,
        help="Período total da análise. Ao final deste prazo, o simulador compara o patrimônio total das duas estratégias."
    )

with st.sidebar.expander("💵 Estratégia: À Vista", expanded=True):
    st.caption(
        "Nesta estratégia, seu caixa fica **rendendo até o ano da compra**. "
        "No ano escolhido, você saca o valor necessário e compra o imóvel à vista. "
        "O restante continua rendendo até o fim do horizonte."
    )
    cash_enabled = st.checkbox(
        "Incluir na comparação", value=True, key="cash_enabled",
        help="Desmarque para remover esta estratégia do gráfico e das tabelas comparativas."
    )
    no_cash_purchase = st.checkbox(
        "Não comprar — só deixar rendendo",
        value=False,
        help="Marque para comparar o consórcio contra a estratégia de simplesmente manter o capital investido, sem comprar imóvel algum. Útil para entender se vale a pena se imobilizar no imóvel.",
        disabled=not cash_enabled,
    )
    if not no_cash_purchase:
        cash_purchase_year = st.number_input(
            "Ano da compra à vista", value=1.0, min_value=0.1, step=0.5, format="%.1f",
            help="Em que ano do horizonte você sacaria o dinheiro para comprar o imóvel à vista. "
                 "Ano 1 = compra imediata (após 12 meses). Quanto maior, mais tempo o caixa rende antes da compra, "
                 "mas o imóvel também fica mais caro pela valorização.",
            disabled=not cash_enabled,
        )
        cash_purchase_extra_cost_pct = st.number_input(
            "Custos extras na compra (%)", value=0.0, min_value=0.0, step=0.1, format="%.2f",
            help="Percentual adicional sobre o preço do imóvel para cobrir ITBI, escritura, registro em cartório, etc. "
                 "Referência típica: 4% a 6% (ITBI ~3% + escritura/registro ~1-2%).",
            disabled=not cash_enabled,
        )
    else:
        cash_purchase_year = float(horizon_years) + 999  # nunca compra
        cash_purchase_extra_cost_pct = 0.0

with st.sidebar.expander("🏦 Estratégia: Consórcio", expanded=True):
    st.caption(
        "No consórcio, você paga **parcelas mensais** e é contemplado em um determinado mês "
        "(aqui, assumimos contemplação **por lance**). Até a contemplação, pode-se pagar uma fração "
        "da parcela cheia. Após o lance, recebe a carta de crédito e compra o imóvel."
    )
    cons_enabled = st.checkbox(
        "Incluir na comparação", value=True, key="cons_enabled",
        help="Desmarque para remover esta estratégia do gráfico e das tabelas comparativas."
    )
    consortium_credit_today = st.number_input(
        "Carta de crédito (R$)", value=0.0, step=50_000.0, min_value=0.0, format="%.2f",
        help="Valor da carta de crédito do consórcio em reais de hoje. "
             "Normalmente igual ou próximo ao preço do imóvel que você quer comprar. "
             "A carta é reajustada anualmente pelo índice pré-contemplação (geralmente INCC).",
        disabled=not cons_enabled,
    )
    term_months = st.number_input(
        "Prazo total do plano (meses)", value=180, min_value=12, max_value=360, step=12,
        help="Duração total do consórcio em meses. Valores comuns: 120, 180, 200, 240. "
             "As parcelas pré-contemplação são proporcionais (carta + taxas ÷ prazo). "
             "Após a contemplação, o saldo devedor é diluído nos meses restantes do plano.",
        disabled=not cons_enabled,
    )
    contemplation_year = st.number_input(
        "Ano de contemplação projetado", value=5.0, min_value=0.1, step=0.5, format="%.1f",
        help="Em que ano você espera ser contemplado (dar o lance vencedor). "
             "Este é o campo mais especulativo — depende do grupo, do valor do lance e da sorte/estratégia. "
             "Teste cenários de contemplação em anos diferentes usando a aba de Sensibilidade.",
        disabled=not cons_enabled,
    )
    pre_contemplation_payment_fraction = st.number_input(
        "Fração da parcela antes da contemplação", value=0.5, min_value=0.01, max_value=1.0, step=0.05, format="%.2f",
        help="Quanto da parcela cheia você paga ANTES de ser contemplado. "
             "Ex: 0,5 = paga metade da parcela cheia. Alguns consórcios permitem parcela reduzida enquanto se aguarda contemplação. "
             "Use 1,0 se for pagar a parcela cheia desde o início.",
        disabled=not cons_enabled,
    )
    bid_pct = st.number_input(
        "Lance total (% da carta)", value=40.0, min_value=0.0, max_value=100.0, step=1.0, format="%.1f",
        help="Percentual do valor do plano oferecido como lance para ser contemplado. "
             "Lances maiores aumentam a chance de contemplação mais cedo, mas exigem mais capital imediato. "
             "No Brasil, lances vencedores costumam ficar entre 30% e 60%, dependendo do grupo.",
        disabled=not cons_enabled,
    )
    embedded_bid_pct_of_bid = st.number_input(
        "Lance embutido (% do lance total)", value=0.0, min_value=0.0, max_value=100.0, step=5.0, format="%.1f",
        help="Parte do lance que é descontada da própria carta de crédito (não sai do seu bolso), "
             "mas em troca você recebe uma carta menor. Ex: carta de R$ 500k, lance total de 30% (R$ 150k), "
             "embutido 50% → R$ 75k sai do bolso, R$ 75k reduz a carta (que vira R$ 425k). "
             "Nem toda administradora permite lance embutido — verifique as regras do seu plano.",
        disabled=not cons_enabled,
    )
    bid_base = st.selectbox(
        "Base do lance", ["adjusted_credit", "original_credit"],
        format_func=lambda x: "Sobre carta reajustada" if x == "adjusted_credit" else "Sobre carta original",
        help="Sobre qual valor o percentual do lance é calculado: a carta já corrigida pelo INCC até o mês da contemplação "
             "(mais comum) ou a carta original de hoje. A maior parte das administradoras usa a carta reajustada.",
        disabled=not cons_enabled,
    )
    reduce_installment_after_bid = st.checkbox(
        "Lance abate saldo devedor futuro", value=True,
        help="Se marcado, o lance reduz o saldo a pagar após a contemplação (diminui as parcelas futuras). "
             "Se desmarcado, o lance apenas serve para contemplar e não reduz o valor total a pagar pelo plano. "
             "A regra depende da administradora — verifique seu contrato.",
        disabled=not cons_enabled,
    )
    admin_fee_pct = st.number_input(
        "Taxa administrativa total (%)", value=20.0, min_value=0.0, step=0.5, format="%.2f",
        help="Taxa da administradora, cobrada sobre o valor da carta e diluída nas parcelas. "
             "No Brasil, costuma ficar entre 15% e 25% do valor da carta. É o principal custo do consórcio.",
        disabled=not cons_enabled,
    )
    reserve_fund_pct = st.number_input(
        "Fundo de reserva (%)", value=1.0, min_value=0.0, step=0.5, format="%.2f",
        help="Fundo comum do grupo para cobrir inadimplência e sorteios extras. "
             "Valor típico: 1% a 3% do valor da carta. Também é diluído nas parcelas.",
        disabled=not cons_enabled,
    )
    adjustment_pre_annual = st.number_input(
        "Reajuste anual pré-contemplação (INCC) (%)", value=5.7, step=0.1, format="%.2f",
        help="Reajuste anual aplicado à carta de crédito e às parcelas ANTES da contemplação. "
             "Geralmente é o INCC (Índice Nacional de Custo da Construção). Histórico recente: entre 4% e 10% a.a.",
        disabled=not cons_enabled,
    )
    adjustment_post_annual = st.number_input(
        "Reajuste anual pós-contemplação (IPCA) (%)", value=5.0, step=0.1, format="%.2f",
        help="Reajuste anual aplicado ao saldo devedor e às parcelas APÓS a contemplação. "
             "Geralmente é o IPCA ou INCC — depende do contrato. Histórico recente do IPCA: 3% a 10% a.a.",
        disabled=not cons_enabled,
    )
    insurance_monthly_pct_of_credit = st.number_input(
        "Seguro mensal (% da carta)", value=0.035, min_value=0.0, step=0.01, format="%.3f",
        help="Seguro de vida/prestamista cobrado mensalmente como percentual da carta de crédito vigente. "
             "Valor típico: 0,03% a 0,05% ao mês. Ex: 0,035% sobre carta de R$ 500 mil = R$ 175/mês. "
             "É um custo adicional além da parcela de amortização.",
        disabled=not cons_enabled,
    )

with st.sidebar.expander("🏛️ Estratégia: Financiamento"):
    st.caption(
        "No financiamento, você paga uma **entrada** no ato (geralmente 20% do imóvel) "
        "e financia o restante por até 30-35 anos. A posse do imóvel é imediata, "
        "mas o banco fica como credor fiduciário até a quitação."
    )
    financing_enabled = st.checkbox(
        "Habilitar estratégia de financiamento", value=False,
        help="Marque para incluir o financiamento bancário na comparação. Quando desmarcado, apenas "
             "as estratégias 'À Vista' e 'Consórcio' são mostradas."
    )
    financing_start_year = st.number_input(
        "Ano da contratação do financiamento", value=1.0, min_value=0.1, step=0.5, format="%.1f",
        help="Em que ano do horizonte você assina o contrato e toma posse do imóvel. "
             "Ano 1 = contratação após 12 meses. A primeira parcela vence no mês seguinte à contratação.",
        disabled=not financing_enabled,
    )
    financing_down_payment_pct = st.number_input(
        "Entrada (% do imóvel)", value=20.0, min_value=0.0, max_value=100.0, step=5.0, format="%.1f",
        help="Percentual do valor do imóvel pago como entrada (recursos próprios). "
             "No SFH os bancos tipicamente financiam até 80% do imóvel, então a entrada mínima "
             "costuma ser 20%. Entradas maiores reduzem juros totais.",
        disabled=not financing_enabled,
    )
    financing_interest_annual = st.number_input(
        "Taxa de juros nominal a.a. (%)", value=11.5, min_value=0.0, step=0.1, format="%.2f",
        help="Taxa de juros efetiva do financiamento (exclui correção monetária, que é campo separado). "
             "Referências 2026: Caixa ~11,2%, Itaú ~11,6%, Santander ~11,7%. MCMV: 4% a 10%. "
             "Taxas pós-fixadas (IPCA+) podem ser menores em número, mas com risco de correção.",
        disabled=not financing_enabled,
    )
    financing_term_months = st.number_input(
        "Prazo do financiamento (meses)", value=360, min_value=12, max_value=420, step=12,
        help="Duração total do financiamento. Prazos comuns: 240 (20 anos), 360 (30 anos) e 420 (35 anos, "
             "máximo permitido pela Caixa). Prazos maiores = parcelas menores mas muito mais juros no total.",
        disabled=not financing_enabled,
    )
    financing_system = st.selectbox(
        "Sistema de amortização", ["SAC", "Price"],
        help="• **SAC** (Sistema de Amortização Constante): amortização mensal fixa, parcelas decrescentes "
             "(começa alta, termina baixa). Paga menos juros no total.\n\n"
             "• **Price** (Tabela Price): parcelas constantes (exceto pelo seguro, que cai com o saldo). "
             "Melhor para quem precisa de previsibilidade, mas paga mais juros no total.\n\n"
             "Em 2026, com juros altos, a literatura recomenda SAC pela economia significativa.",
        disabled=not financing_enabled,
    )
    financing_extra_cost_pct = st.number_input(
        "Custos extras de aquisição (%)", value=5.0, min_value=0.0, step=0.1, format="%.2f",
        help="Percentual sobre o valor do imóvel para ITBI (~3%), escritura e registro em cartório (~1-2%), "
             "avaliação do imóvel e outras taxas. Referência: 4% a 6%. "
             "Esse valor sai do caixa no ato da contratação, junto com a entrada.",
        disabled=not financing_enabled,
    )
    financing_insurance_monthly_pct_of_balance = st.number_input(
        "Seguro mensal (% do saldo devedor)", value=0.035, min_value=0.0, step=0.005, format="%.3f",
        help="Seguro MIP (morte/invalidez) + DFI (danos físicos ao imóvel), cobrado mensalmente "
             "como percentual do saldo devedor. Cai ao longo do tempo conforme o saldo diminui. "
             "Valor típico: 0,025% a 0,05% ao mês.",
        disabled=not financing_enabled,
    )
    financing_balance_adjustment_annual = st.number_input(
        "Correção anual do saldo devedor (%)", value=1.5, min_value=0.0, step=0.1, format="%.2f",
        help="Índice de correção aplicado anualmente ao saldo devedor, além dos juros. "
             "Na maioria dos contratos Caixa é a **TR** (historicamente entre 0% e 2% a.a.). "
             "Contratos IPCA+ usam o IPCA (3% a 10% a.a.). Deixe 0 para taxa puramente fixa.",
        disabled=not financing_enabled,
    )
    st.markdown("---")
    st.markdown("**Amortização antecipada (opcional)**")
    st.caption(
        "Permite aportes extras periódicos no saldo devedor. "
        "⚠️ Se o caixa não comportar o valor no mês agendado, a amortização é **pulada** "
        "(não ocorre parcialmente)."
    )
    prepayment_amount = st.number_input(
        "Valor da amortização (R$)", value=0.0, min_value=0.0, step=5_000.0, format="%.2f",
        help="Valor em reais de cada aporte extra no saldo devedor. Sai do caixa investido. "
             "Deixe 0 para não usar amortização antecipada.",
        disabled=not financing_enabled,
    )
    prepayment_period_months = st.number_input(
        "Periodicidade (meses)", value=0, min_value=0, max_value=120, step=1,
        help="A cada quantos meses ocorre uma amortização. Ex: 12 = uma vez por ano, 6 = semestralmente. "
             "A contagem começa a partir do mês da contratação do financiamento. "
             "Deixe 0 para não usar amortização antecipada.",
        disabled=not financing_enabled,
    )
    prepayment_adjustment_annual = st.number_input(
        "Correção anual do valor (%)", value=0.0, min_value=0.0, step=0.1, format="%.2f",
        help="Ajuste anual aplicado ao valor da amortização (para acompanhar inflação / renda). "
             "Ex: 5% = amortização de R$ 10k no ano 1 vira R$ 10,5k no ano 2, etc. "
             "Útil quando a amortização vem de renda que cresce com o tempo. Deixe 0 para valor constante.",
        disabled=not financing_enabled,
    )
    prepayment_mode = st.selectbox(
        "Política da amortização",
        ["reduce_term", "reduce_installment"],
        format_func=lambda x: "Reduzir prazo (quita antes)" if x == "reduce_term"
                              else "Reduzir parcela (paga menos por mês)",
        help="• **Reduzir prazo**: mantém o valor da parcela e antecipa a quitação. "
             "É a opção mais vantajosa financeiramente (economiza mais juros).\n\n"
             "• **Reduzir parcela**: mantém o prazo original e diminui a parcela mensal. "
             "Melhora o fluxo de caixa mensal, mas economiza menos juros no total.",
        disabled=not financing_enabled,
    )

with st.sidebar.expander("💰 Poupança Mensal", expanded=True):
    st.caption(
        "Valor que você aporta mensalmente no caixa — **aplicado igualmente nas duas estratégias** "
        "para garantir comparação justa."
    )
    monthly_savings = st.number_input(
        "Aporte mensal (R$)", value=0.0, min_value=0.0, step=500.0, format="%.2f",
        help="Valor depositado todo mês no caixa, em ambas as estratégias. "
             "Use para simular renda extra destinada ao investimento. Deixe em 0 para não considerar aportes."
    )
    savings_inflation_annual = st.number_input(
        "Correção do aporte a.a. (%)", value=5.0, step=0.1, format="%.2f",
        help="O aporte cresce anualmente por este percentual (ex: IPCA para manter poder de compra). "
             "Representa seu aumento de renda ao longo do tempo."
    )

with st.sidebar.expander("🏠 Pós-compra (todas as estratégias)"):
    st.caption(
        "Efeitos financeiros que passam a valer **após a posse do imóvel** — em todas as estratégias, "
        "a partir do mês da compra (à vista), contemplação (consórcio) ou contratação (financiamento)."
    )
    monthly_rent_saved_after_purchase = st.number_input(
        "Aluguel evitado ou recebido após posse (R$/mês)", value=0.0, min_value=0.0, step=500.0, format="%.2f",
        help="Entrada líquida mensal de caixa após tomar posse do imóvel. Representa um destes casos "
             "(são mutuamente exclusivos — use apenas um):\n\n"
             "• **Aluguel evitado** — cliente morava de aluguel e passará a morar no imóvel comprado.\n"
             "• **Aluguel recebido** — imóvel é para investimento/locação e será alugado a terceiros.\n\n"
             "Em ambos os casos, entra como receita mensal no caixa a partir da posse. Deixe 0 se não se aplica "
             "(ex: cliente já era proprietário de outro imóvel e não vai alugar este)."
    )
    rent_inflation_annual = st.number_input(
        "Inflação do aluguel a.a. (%)", value=5.0, step=0.1, format="%.2f",
        help="Reajuste anual do aluguel (evitado ou recebido). Geralmente IGPM ou IPCA. "
             "Histórico recente: 3% a 10% a.a."
    )
    vacancy_months_initial = st.number_input(
        "Vacância inicial (meses)", value=0, min_value=0, max_value=60, step=1,
        help="Tempo estimado, em meses, até conseguir efetivamente o aluguel após a posse do imóvel. "
             "Durante esse período, o 'aluguel evitado ou recebido' é considerado **zero** "
             "— o imóvel está vazio aguardando inquilino ou reforma.\n\n"
             "Usar quando:\n"
             "• O imóvel é para investimento e leva X meses para encontrar locatário.\n"
             "• O cliente planeja fazer reformas antes de mudar ou alugar.\n"
             "• Precisa de tempo para sair do aluguel atual (aviso prévio, mudança etc.).\n\n"
             "Referência de mercado: imóveis residenciais urbanos costumam ter 1-3 meses "
             "de vacância entre inquilinos. Se não se aplica, deixe em 0."
    )
    monthly_property_cost_after_purchase = st.number_input(
        "Custo mensal do imóvel após compra (R$)", value=0.0, min_value=0.0, step=100.0, format="%.2f",
        help="Custos fixos de ter o imóvel: IPTU mensalizado, condomínio, manutenção, seguro residencial, etc. "
             "Entra como saída mensal do caixa após a compra (inclusive durante a vacância)."
    )

with st.sidebar.expander("🔧 Avançado"):
    monthly_precision_rounding = st.checkbox(
        "Arredondar centavos mensalmente", value=False,
        help="Arredonda o saldo para 2 casas decimais a cada mês (elimina diferenças de floating point). "
             "Só faz diferença perceptível em horizontes muito longos. Deixe desmarcado para máxima precisão."
    )
    st.markdown("**Informações de IR (apenas para auditoria)**")
    st.caption(
        "Os campos abaixo **não afetam a simulação** — a taxa de rendimento líquida já foi informada em 'Gerais'. "
        "Servem apenas para estimar quanto IR você teria pagado, para fins de auditoria."
    )
    ir_aliquota_pct = st.number_input(
        "Alíquota de IR sobre rendimentos (%)", value=15.0, min_value=0.0, max_value=100.0, step=0.5, format="%.1f",
        help="Usado apenas para estimar o IR pago — não afeta a simulação. "
             "Referências: isento (LCI/LCA) = 0%; >720 dias = 15%; 361-720 dias = 17,5%; 181-360 dias = 20%; até 180 dias = 22,5%."
    )
    investment_return_annual_gross = st.number_input(
        "Rentabilidade bruta a.a. (%) — referência", value=13.75, step=0.1, format="%.2f",
        help="Usado apenas para calcular o IR estimado na auditoria. Não afeta a simulação. "
             "Fórmula aproximada: líquida ≈ bruta × (1 − alíquota IR)."
    )

# ---- SIMULAÇÃO ----
try:
    inp = Inputs(
        initial_cash=initial_cash,
        property_price_today=property_price_today,
        property_appreciation_annual=property_appreciation_annual,
        investment_return_annual_net=investment_return_annual_net,
        horizon_years=int(horizon_years),
        cash_purchase_year=cash_purchase_year,
        cash_purchase_extra_cost_pct=cash_purchase_extra_cost_pct,
        consortium_credit_today=consortium_credit_today,
        admin_fee_pct=admin_fee_pct,
        reserve_fund_pct=reserve_fund_pct,
        insurance_monthly_pct_of_credit=insurance_monthly_pct_of_credit,
        term_months=int(term_months),
        contemplation_year=contemplation_year,
        pre_contemplation_payment_fraction=pre_contemplation_payment_fraction,
        bid_pct=bid_pct,
        embedded_bid_pct_of_bid=embedded_bid_pct_of_bid,
        bid_base=bid_base,
        adjustment_pre_annual=adjustment_pre_annual,
        adjustment_post_annual=adjustment_post_annual,
        reduce_installment_after_bid=reduce_installment_after_bid,
        monthly_rent_saved_after_purchase=monthly_rent_saved_after_purchase,
        rent_inflation_annual=rent_inflation_annual,
        vacancy_months_initial=int(vacancy_months_initial),
        monthly_property_cost_after_purchase=monthly_property_cost_after_purchase,
        monthly_precision_rounding=monthly_precision_rounding,
        monthly_savings=monthly_savings,
        savings_inflation_annual=savings_inflation_annual,
        ir_aliquota_pct=ir_aliquota_pct,
        investment_return_annual_gross=investment_return_annual_gross,
        no_cash_purchase=no_cash_purchase,
        cash_enabled=cash_enabled,
        cons_enabled=cons_enabled,
        # Financiamento
        financing_enabled=financing_enabled,
        financing_start_year=financing_start_year,
        financing_down_payment_pct=financing_down_payment_pct,
        financing_interest_annual=financing_interest_annual,
        financing_term_months=int(financing_term_months),
        financing_system=financing_system,
        financing_extra_cost_pct=financing_extra_cost_pct,
        financing_insurance_monthly_pct_of_balance=financing_insurance_monthly_pct_of_balance,
        financing_balance_adjustment_annual=financing_balance_adjustment_annual,
        prepayment_amount=prepayment_amount,
        prepayment_period_months=int(prepayment_period_months),
        prepayment_adjustment_annual=prepayment_adjustment_annual,
        prepayment_mode=prepayment_mode,
    )

    # Checa campos obrigatórios não preenchidos (= 0)
    missing = missing_required_fields(inp)
    if missing:
        st.info(
            "👋 **Bem-vindo!** Para começar a simulação, preencha os seguintes campos "
            "na barra lateral à esquerda:\n\n"
            + "\n".join(f"- **{campo}**" for campo in missing)
            + "\n\nDica: valores típicos no Brasil são de R$ 300k a R$ 2M para imóveis residenciais urbanos."
        )
        st.stop()

    errors = validate_inputs(inp)
    if errors:
        for e in errors:
            st.error(f"⚠️ {e}")
        st.stop()

    # Guard: pelo menos uma estratégia precisa estar incluída
    if not (inp.cash_enabled or inp.cons_enabled or inp.financing_enabled):
        st.warning(
            "👀 **Nenhuma estratégia selecionada.** Marque ao menos uma das caixas "
            "**'Incluir na comparação'** nas estratégias (À Vista, Consórcio ou Financiamento) "
            "na barra lateral para iniciar a simulação."
        )
        st.stop()

    # Roda só as simulações das estratégias habilitadas — evita trabalho desnecessário
    cash_result = simulate_cash_later(inp) if inp.cash_enabled else None
    cons_result = simulate_consortium(inp) if inp.cons_enabled else None
    fin_result = simulate_financing(inp) if inp.financing_enabled else None

    # Alertas de saldo negativo
    if cash_result is not None and cash_result["went_negative"]:
        st.warning("⚠️ **Compra à Vista:** o saldo financeiro ficou negativo em algum momento da simulação. "
                   "Considere aumentar o caixa inicial ou o aporte mensal.")
    if cons_result is not None and cons_result["went_negative"]:
        st.warning("⚠️ **Consórcio:** o saldo financeiro ficou negativo em algum momento da simulação. "
                   "Considere aumentar o caixa inicial ou o aporte mensal.")
    if fin_result is not None and fin_result["went_negative"]:
        st.warning("⚠️ **Financiamento:** o saldo financeiro ficou negativo em algum momento da simulação. "
                   "As parcelas podem estar muito altas para o caixa disponível — considere aumentar "
                   "a entrada, o aporte mensal, ou escolher um prazo maior.")

    label_cash = "Só Rendendo" if inp.no_cash_purchase else "Compra à Vista"

    # Determinar a estratégia vencedora (considera apenas as incluídas na comparação)
    candidates = []
    if cash_result is not None:
        candidates.append((label_cash, cash_result["final_total_wealth"], "winner-cash"))
    if cons_result is not None:
        candidates.append(("Consórcio", cons_result["final_total_wealth"], "winner-consortium"))
    if fin_result is not None:
        candidates.append(("Financiamento", fin_result["final_total_wealth"], "winner-financing"))

    # Ordena por patrimônio decrescente
    candidates.sort(key=lambda c: c[1], reverse=True)
    winner_label, winner_value, winner_class = candidates[0]

    if len(candidates) == 1:
        # Só uma estratégia incluída — não há comparativo
        st.markdown(
            f'<div class="winner-box {winner_class}">'
            f'📍 Estratégia em análise: <strong>{winner_label}</strong> '
            f'— patrimônio total final de <strong>{fmt_brl(winner_value)}</strong>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        runner_up = candidates[1]
        advantage = winner_value - runner_up[1]
        st.markdown(
            f'<div class="winner-box {winner_class}">'
            f'🏆 Estratégia vencedora: <strong>{winner_label}</strong> '
            f'— vantagem de <strong>{fmt_brl(abs(advantage))}</strong> sobre a segunda colocada '
            f'({runner_up[0]}) ao final do período'
            f'</div>',
            unsafe_allow_html=True
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Gráfico", "📝 Resumo", "🔎 Auditoria", "📅 Ano a Ano", "📉 Sensibilidade"])

    # ---- GRÁFICO ----
    with tab1:
        # Monta legenda dinamicamente
        legend_parts = []
        events_parts = []
        if cash_result is not None:
            legend_parts.append("🟦 Azul = À Vista")
            if not inp.no_cash_purchase:
                events_parts.append("compra à vista 🔵")
        if cons_result is not None:
            legend_parts.append("🟧 Laranja = Consórcio")
            events_parts.append("contemplação 🟠")
        if fin_result is not None:
            legend_parts.append("🟩 Verde = Financiamento")
            events_parts.append("contratação do financiamento 🟢")

        events_text = ", ".join(events_parts) if events_parts else "(nenhum evento a marcar)"
        legend_text = " | ".join(legend_parts)

        st.markdown(
            "### Evolução do Patrimônio ao Longo do Tempo\n"
            "As **linhas sólidas** mostram o patrimônio total (caixa + valor do imóvel − dívida). "
            "As **linhas pontilhadas** mostram apenas o caixa financeiro. "
            f"As **linhas verticais tracejadas** marcam os eventos-chave: {events_text}.\n\n"
            + legend_text
        )

        fig = go.Figure()

        # Linhas de cada estratégia (só as habilitadas)
        if cash_result is not None:
            df_cash = pd.DataFrame(cash_result["records"])
            fig.add_trace(go.Scatter(
                x=df_cash["year"], y=df_cash["total_wealth"],
                name=f"{label_cash} — Total", mode="lines",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="Ano %{x:.1f}<br>Total: R$ %{y:,.0f}<extra>" + label_cash + "</extra>"
            ))
            fig.add_trace(go.Scatter(
                x=df_cash["year"], y=df_cash["financial_wealth"],
                name=f"{label_cash} — Caixa", mode="lines",
                line=dict(color="#1f77b4", width=1, dash="dot"),
                hovertemplate="Ano %{x:.1f}<br>Caixa: R$ %{y:,.0f}<extra>" + label_cash + "</extra>"
            ))

        if cons_result is not None:
            df_cons = pd.DataFrame(cons_result["records"])
            fig.add_trace(go.Scatter(
                x=df_cons["year"], y=df_cons["total_wealth"],
                name="Consórcio — Total", mode="lines",
                line=dict(color="#ff7f0e", width=2.5),
                hovertemplate="Ano %{x:.1f}<br>Total: R$ %{y:,.0f}<extra>Consórcio</extra>"
            ))
            fig.add_trace(go.Scatter(
                x=df_cons["year"], y=df_cons["financial_wealth"],
                name="Consórcio — Caixa", mode="lines",
                line=dict(color="#ff7f0e", width=1, dash="dot"),
                hovertemplate="Ano %{x:.1f}<br>Caixa: R$ %{y:,.0f}<extra>Consórcio</extra>"
            ))

        if fin_result is not None:
            df_fin = pd.DataFrame(fin_result["records"])
            fig.add_trace(go.Scatter(
                x=df_fin["year"], y=df_fin["total_wealth"],
                name="Financiamento — Total", mode="lines",
                line=dict(color="#2ca02c", width=2.5),
                hovertemplate="Ano %{x:.1f}<br>Total: R$ %{y:,.0f}<extra>Financiamento</extra>"
            ))
            fig.add_trace(go.Scatter(
                x=df_fin["year"], y=df_fin["financial_wealth"],
                name="Financiamento — Caixa", mode="lines",
                line=dict(color="#2ca02c", width=1, dash="dot"),
                hovertemplate="Ano %{x:.1f}<br>Caixa: R$ %{y:,.0f}<extra>Financiamento</extra>"
            ))

        # Linhas verticais de eventos
        if cash_result is not None and not inp.no_cash_purchase:
            fig.add_vline(
                x=inp.cash_purchase_year, line_dash="dash", line_color="#1f77b4",
                annotation_text=f"Compra à Vista (Ano {inp.cash_purchase_year:.1f})",
                annotation_position="top left", annotation_font_size=11
            )
        if cons_result is not None:
            fig.add_vline(
                x=inp.contemplation_year, line_dash="dash", line_color="#ff7f0e",
                annotation_text=f"Contemplação (Ano {inp.contemplation_year:.1f})",
                annotation_position="top right", annotation_font_size=11
            )
        if fin_result is not None:
            fig.add_vline(
                x=inp.financing_start_year, line_dash="dash", line_color="#2ca02c",
                annotation_text=f"Contratação Financ. (Ano {inp.financing_start_year:.1f})",
                annotation_position="bottom right", annotation_font_size=11
            )

        fig.update_layout(
            xaxis_title="Ano",
            yaxis_title="Patrimônio (R$)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            height=480,
            margin=dict(t=60, b=40),
            yaxis=dict(tickformat=",.0f", tickprefix="R$ "),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Gráfico de diferença — só faz sentido se houver uma referência (À Vista) + ao menos outra
        if cash_result is not None and (cons_result is not None or fin_result is not None):
            st.markdown(
                f"#### Diferença de Patrimônio vs {label_cash}\n"
                "Barras **acima de zero** indicam vantagem da estratégia em relação à opção à vista naquele ano. "
                "Barras **abaixo de zero** indicam desvantagem. "
                "Observe o ano em que cada estratégia cruza a linha do zero — esse costuma ser o ponto decisivo."
            )

            years_series = [r["year"] for r in cash_result["records"]]
            fig2 = go.Figure()

            if cons_result is not None:
                diff_cons = [
                    c["total_wealth"] - a["total_wealth"]
                    for c, a in zip(cons_result["records"], cash_result["records"])
                ]
                fig2.add_trace(go.Bar(
                    x=years_series, y=diff_cons,
                    marker_color="#ff7f0e",
                    name="Consórcio",
                    hovertemplate="Ano %{x:.1f}<br>Consórcio vs À Vista: R$ %{y:,.0f}<extra></extra>"
                ))
            if fin_result is not None:
                diff_fin = [
                    f["total_wealth"] - a["total_wealth"]
                    for f, a in zip(fin_result["records"], cash_result["records"])
                ]
                fig2.add_trace(go.Bar(
                    x=years_series, y=diff_fin,
                    marker_color="#2ca02c",
                    name="Financiamento",
                    hovertemplate="Ano %{x:.1f}<br>Financ. vs À Vista: R$ %{y:,.0f}<extra></extra>"
                ))
            fig2.add_hline(y=0, line_color="gray", line_width=1)
            fig2.update_layout(
                title=f"Diferença de Patrimônio Total em relação a {label_cash}",
                xaxis_title="Ano",
                yaxis_title="Diferença (R$)",
                yaxis=dict(tickformat=",.0f", tickprefix="R$ "),
                height=320,
                barmode="group",
                margin=dict(t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig2, use_container_width=True)
        elif cash_result is None:
            st.info(
                "ℹ️ O gráfico de diferença compara as estratégias contra a **Compra à Vista** (linha de base). "
                "Habilite a estratégia **À Vista** na barra lateral para visualizá-lo."
            )

    # ---- RESUMO ----
    with tab2:
        # Conta quantas estratégias estão ativas
        active_results = []
        if cash_result is not None:
            active_results.append(("cash", label_cash, "💵", cash_result))
        if cons_result is not None:
            active_results.append(("cons", "Consórcio", "🏦", cons_result))
        if fin_result is not None:
            active_results.append(("fin", "Financiamento", "🏛️", fin_result))
        n_active = len(active_results)

        n_text = {1: "uma", 2: "duas", 3: "três"}[n_active]
        st.markdown(
            "### Posição Final no Horizonte\n"
            f"Compare os resultados da{'s' if n_active > 1 else ''} {n_text} estratégia{'s' if n_active > 1 else ''} "
            "ao fim do período. "
            "**Patrimônio Total** é o que você teria se vendesse tudo no último dia: "
            "caixa investido + valor de mercado do imóvel − dívidas residuais. "
            "**Desembolso Total do Bolso** é quanto saiu efetivamente do seu caixa ao longo da simulação."
        )

        # Layout em colunas conforme o número de estratégias ativas
        cols = st.columns(n_active)

        for (key, name, emoji, result), col in zip(active_results, cols):
            with col:
                st.markdown(f"### {emoji} {name}")
                st.metric("Patrimônio Total Final", fmt_brl(result["final_total_wealth"]))
                st.metric("Patrimônio Financeiro (Caixa)", fmt_brl(result["final_financial_wealth"]))
                st.metric("Patrimônio Imobiliário", fmt_brl(result["final_real_estate_wealth"]))

                if key == "fin" and result["final_remaining_debt"] > 0:
                    st.metric(
                        "Saldo Devedor Residual", f"-{fmt_brl(result['final_remaining_debt'])}",
                        help="Quanto ainda resta pagar do financiamento ao fim do horizonte"
                    )

                st.metric("Desembolso Total do Bolso", fmt_brl(result["out_of_pocket_total"]))

                # Métricas específicas do consórcio
                if key == "cons":
                    if result["final_cumulative_property_topup"] > 0:
                        st.metric(
                            "Complemento próprio p/ cobrir imóvel",
                            fmt_brl(result["final_cumulative_property_topup"]),
                            help="Valor tirado do caixa porque a carta líquida (após lance embutido) não cobriu o preço do imóvel"
                        )
                    if result["final_cumulative_bid_own_cash"] > 0:
                        st.metric("Lance do próprio bolso", fmt_brl(result["final_cumulative_bid_own_cash"]))
                    if result["final_cumulative_bid_embedded"] > 0:
                        st.metric("Lance embutido (da carta)", fmt_brl(result["final_cumulative_bid_embedded"]))

                # Métricas específicas do financiamento
                if key == "fin":
                    st.metric(
                        "Juros Totais Pagos", fmt_brl(result["final_cumulative_interest"]),
                        help="Soma dos juros pagos em todas as parcelas"
                    )
                    if result["final_cumulative_prepayments"] > 0:
                        st.metric(
                            "Amortizações Antecipadas", fmt_brl(result["final_cumulative_prepayments"]),
                            help="Total aportado em amortizações extraordinárias"
                        )

                st.metric("IR pago sobre rendimentos", fmt_brl(result["cumulative_ir_paid"]))
                if inp.monthly_savings > 0:
                    st.metric("Total aportado no período", fmt_brl(result["cumulative_savings"]))

        # Tabela comparativa — só se tiver pelo menos 2 estratégias
        if n_active >= 2:
            st.divider()
            st.markdown("### 📊 Comparativo Final")

            # Constrói dict de colunas dinamicamente
            metrics_base = ["Patrimônio Total", "Patrimônio Financeiro", "Patrimônio Imobiliário", "Desembolso Total", "IR pago (rendimentos)"]

            def base_values(result):
                return [
                    result["final_total_wealth"],
                    result["final_financial_wealth"],
                    result["final_real_estate_wealth"],
                    result["out_of_pocket_total"],
                    result["cumulative_ir_paid"],
                ]

            # Por estratégia, lista de valores
            data_by_strategy: dict[str, list[float]] = {}
            for key, name, emoji, result in active_results:
                data_by_strategy[name] = base_values(result)

            metrics = list(metrics_base)

            # Linhas específicas do consórcio (só se consórcio ativo)
            if cons_result is not None:
                extras_cons = [
                    ("Complemento próprio p/ imóvel", cons_result["final_cumulative_property_topup"]),
                    ("Lance do próprio bolso", cons_result["final_cumulative_bid_own_cash"]),
                    ("Lance embutido (da carta)", cons_result["final_cumulative_bid_embedded"]),
                ]
                for metric_name, value in extras_cons:
                    if value > 0:
                        metrics.append(metric_name)
                        for strat_name in data_by_strategy:
                            data_by_strategy[strat_name].append(
                                value if strat_name == "Consórcio" else 0.0
                            )

            # Linhas específicas do financiamento (só se financiamento ativo)
            if fin_result is not None:
                extras_fin = [
                    ("Juros totais pagos", fin_result["final_cumulative_interest"]),
                    ("Dívida residual (−)", fin_result["final_remaining_debt"]),
                    ("Amortizações antecipadas", fin_result["final_cumulative_prepayments"]),
                ]
                for metric_name, value in extras_fin:
                    if value > 0:
                        metrics.append(metric_name)
                        for strat_name in data_by_strategy:
                            data_by_strategy[strat_name].append(
                                value if strat_name == "Financiamento" else 0.0
                            )

            # Monta DataFrame
            comp_df_data = {"Métrica": metrics}
            for strat_name, values in data_by_strategy.items():
                comp_df_data[strat_name] = [fmt_brl(v) for v in values]

            # Adiciona coluna de diferença quando for exatamente 2 estratégias (e uma delas é À Vista)
            if n_active == 2 and cash_result is not None:
                other_name = next(name for name in data_by_strategy if name != label_cash)
                cash_values = data_by_strategy[label_cash]
                other_values = data_by_strategy[other_name]
                comp_df_data[f"Diferença ({other_name} − {label_cash})"] = [
                    fmt_brl(o - c) for o, c in zip(other_values, cash_values)
                ]

            comp_df = pd.DataFrame(comp_df_data)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ---- AUDITORIA ----
    with tab3:
        if cons_result is None and fin_result is None:
            st.info(
                "ℹ️ Não há dados de auditoria para exibir. A auditoria detalha a mecânica do "
                "**Consórcio** e/ou do **Financiamento** — habilite ao menos uma dessas estratégias "
                "na barra lateral para ver os detalhes."
            )

        if cons_result is not None:
            st.markdown("### 🔎 Dados de Auditoria do Consórcio")
            st.markdown(
                "Esta seção detalha como os valores do consórcio foram calculados, para você conferir a mecânica:\n\n"
                "- **Saldo devedor na contemplação**: quanto ainda faltaria pagar do plano logo após a contemplação "
                "(já considerando o lance, se ele abate saldo futuro).\n"
                "- **Carta efetiva recebida**: o valor da carta ajustado, descontando o lance embutido "
                "(se houver). É o que você de fato usa para comprar o imóvel.\n"
                "- **Saldo devedor residual**: o que ainda resta do plano ao fim do horizonte da simulação.\n"
                "- **Complemento do imóvel**: dinheiro adicional do seu bolso caso a carta efetiva seja menor "
                "que o preço do imóvel corrigido."
            )

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Contemplação**")
                st.metric("Saldo devedor na contemplação", fmt_brl(cons_result["debt_at_contemplation"]))
                st.metric("Carta efetiva recebida na contemplação", fmt_brl(cons_result["effective_credit_at_contemplation"]))
                st.metric("Saldo devedor residual no fim do horizonte", fmt_brl(cons_result["final_remaining_debt"]))

            with col_b:
                st.markdown("**Desembolsos Acumulados**")
                st.metric("Total de parcelas pagas", fmt_brl(cons_result["final_cumulative_installments"]))
                st.metric("Total de seguro pago", fmt_brl(cons_result["final_cumulative_insurance"]))
                st.metric("Lance do bolso", fmt_brl(cons_result["final_cumulative_bid_own_cash"]))
                st.metric("Lance embutido (deduzido da carta)", fmt_brl(cons_result["final_cumulative_bid_embedded"]))
                st.metric("Complemento do imóvel (bolso)", fmt_brl(cons_result["final_cumulative_property_topup"]))

            st.markdown("**Decomposição do desembolso total no consórcio**")
            decomp = {
                "Parcelas": cons_result["final_cumulative_installments"],
                "Seguro": cons_result["final_cumulative_insurance"],
                "Lance (bolso)": cons_result["final_cumulative_bid_own_cash"],
                "Complemento imóvel": cons_result["final_cumulative_property_topup"],
            }
            # Filtra zeros para não mostrar fatias vazias no gráfico
            decomp = {k: v for k, v in decomp.items() if v > 0}
            if decomp:
                fig3 = px.pie(
                    names=list(decomp.keys()),
                    values=list(decomp.values()),
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig3.update_traces(textinfo="label+percent", hovertemplate="%{label}: R$ %{value:,.0f}<extra></extra>")
                fig3.update_layout(height=350, margin=dict(t=20, b=20))
                st.plotly_chart(fig3, use_container_width=True)

        # Auditoria do financiamento (quando habilitado)
        if fin_result is not None:
            if cons_result is not None:
                st.divider()
            st.markdown("### 🔎 Dados de Auditoria do Financiamento")
            st.markdown(
                "Detalhamento da mecânica do financiamento bancário:\n\n"
                "- **Entrada paga**: valor pago no ato da contratação (% sobre o preço do imóvel corrigido).\n"
                "- **Custos extras de aquisição**: ITBI + cartório + registro + avaliação, pagos no ato.\n"
                "- **Valor financiado**: preço do imóvel − entrada. É o saldo inicial da dívida.\n"
                "- **Juros totais pagos**: soma de todos os juros pagos ao longo das parcelas.\n"
                "- **Saldo devedor residual**: o que ainda resta da dívida ao fim do horizonte "
                "(se o prazo do financiamento ultrapassa o horizonte da análise)."
            )

            col_fa, col_fb = st.columns(2)

            with col_fa:
                st.markdown("**Contratação**")
                st.metric("Entrada paga (no ato)", fmt_brl(fin_result["down_payment"]))
                st.metric("Custos extras de aquisição", fmt_brl(fin_result["acquisition_extra"]))
                st.metric("Valor financiado inicial", fmt_brl(fin_result["financed_amount"]))
                st.metric("Saldo devedor residual (fim do horizonte)", fmt_brl(fin_result["final_remaining_debt"]))

            with col_fb:
                st.markdown("**Desembolsos Acumulados**")
                st.metric("Total de parcelas pagas (princ. + juros)", fmt_brl(fin_result["final_cumulative_installments"]))
                st.metric("Dos quais juros", fmt_brl(fin_result["final_cumulative_interest"]))
                st.metric("Total de seguro pago", fmt_brl(fin_result["final_cumulative_insurance"]))
                st.metric("Amortização antecipada acumulada", fmt_brl(fin_result["final_cumulative_prepayments"]))

            st.markdown("**Decomposição do desembolso total no financiamento**")
            decomp_fin = {
                "Entrada": fin_result["down_payment"],
                "Custos de aquisição": fin_result["acquisition_extra"],
                "Juros das parcelas": fin_result["final_cumulative_interest"],
                "Amortização das parcelas": fin_result["final_cumulative_amortization"] - fin_result["final_cumulative_prepayments"],
                "Seguro": fin_result["final_cumulative_insurance"],
                "Amort. antecipadas": fin_result["final_cumulative_prepayments"],
            }
            # Filtra zeros
            decomp_fin = {k: v for k, v in decomp_fin.items() if v > 0}
            fig_fin = px.pie(
                names=list(decomp_fin.keys()),
                values=list(decomp_fin.values()),
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_fin.update_traces(textinfo="label+percent", hovertemplate="%{label}: R$ %{value:,.0f}<extra></extra>")
            fig_fin.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig_fin, use_container_width=True)

    # ---- ANO A ANO ----
    with tab4:
        st.markdown(
            "### 📅 Evolução Anual do Patrimônio\n"
            "Tabela com a foto do patrimônio ao final de cada ano, em cada estratégia habilitada. "
            "Útil para identificar o **ano do cruzamento** (quando uma estratégia passa à frente da outra) "
            "e para exportar os dados para análise externa. As colunas **Parcela Média** mostram a média "
            "dos 12 meses do ano em consórcio/financiamento — valores menores em anos iniciais "
            "(parcelas pré-contemplação reduzidas no consórcio, ou saldo inicial menor após correção no financiamento)."
        )

        # Parcela média do consórcio/financiamento no ano: média dos 12 meses anteriores
        def avg_installment_for_year(records, year_idx):
            """Média da parcela mensal nos 12 meses do ano."""
            month_end = year_idx * 12
            month_start = month_end - 11
            if month_start < 0:
                return 0.0
            monthly = [r["monthly_outflow"] for r in records if month_start <= r["month"] <= month_end]
            return sum(monthly) / len(monthly) if monthly else 0.0

        # Usa qualquer dos results para pegar a grade de anos (todos têm a mesma)
        base_records = None
        for r in (cash_result, cons_result, fin_result):
            if r is not None:
                base_records = r["records"]
                break
        annual_base = [r for r in base_records if r["month"] % 12 == 0]

        df_data = {"Ano": [int(r["year"]) for r in annual_base]}

        if cash_result is not None:
            cash_annual = [r for r in cash_result["records"] if r["month"] % 12 == 0]
            df_data[f"{label_cash}: Caixa"] = [fmt_brl(r["financial_wealth"]) for r in cash_annual]
            df_data[f"{label_cash}: Imóvel"] = [fmt_brl(r["real_estate_wealth"]) for r in cash_annual]
            df_data[f"{label_cash}: Total"] = [fmt_brl(r["total_wealth"]) for r in cash_annual]

        if cons_result is not None:
            cons_annual = [r for r in cons_result["records"] if r["month"] % 12 == 0]
            df_data["Consórcio: Parcela Média"] = [
                fmt_brl(avg_installment_for_year(cons_result["records"], int(r["year"])))
                for r in cons_annual
            ]
            df_data["Consórcio: Caixa"] = [fmt_brl(r["financial_wealth"]) for r in cons_annual]
            df_data["Consórcio: Imóvel"] = [fmt_brl(r["real_estate_wealth"]) for r in cons_annual]
            df_data["Consórcio: Total"] = [fmt_brl(r["total_wealth"]) for r in cons_annual]

        if fin_result is not None:
            fin_annual = [r for r in fin_result["records"] if r["month"] % 12 == 0]
            df_data["Financ.: Parcela Média"] = [
                fmt_brl(avg_installment_for_year(fin_result["records"], int(r["year"])))
                for r in fin_annual
            ]
            df_data["Financ.: Caixa"] = [fmt_brl(r["financial_wealth"]) for r in fin_annual]
            df_data["Financ.: Imóvel"] = [fmt_brl(r["real_estate_wealth"]) for r in fin_annual]
            df_data["Financ.: Dívida"] = [fmt_brl(-r["remaining_debt"]) for r in fin_annual]
            df_data["Financ.: Total"] = [fmt_brl(r["total_wealth"]) for r in fin_annual]

        # Colunas de diferença (apenas com À Vista como base, e só quando há 2+ estratégias)
        if cash_result is not None:
            if cons_result is not None:
                df_data["Dif. (Cons − AV)"] = [
                    fmt_brl(c["total_wealth"] - a["total_wealth"])
                    for c, a in zip(cons_annual, cash_annual)
                ]
            if fin_result is not None:
                df_data["Dif. (Financ. − AV)"] = [
                    fmt_brl(f["total_wealth"] - a["total_wealth"])
                    for f, a in zip(fin_annual, cash_annual)
                ]

        df_years = pd.DataFrame(df_data)
        st.dataframe(df_years, hide_index=True, use_container_width=True)

        csv = df_years.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar tabela como CSV",
            data=csv,
            file_name="simulacao_estrategias_imoveis.csv",
            mime="text/csv",
        )

    # ---- SENSIBILIDADE ----
    with tab5:
        st.markdown("### 📉 Análise de Sensibilidade")
        st.markdown(
            "Descubra **quais premissas mais influenciam o resultado**. "
            "Escolha um parâmetro para variar e o simulador roda dezenas de cenários mudando apenas ele, "
            "mantendo todos os outros fixos. Útil para responder perguntas como:\n\n"
            "- *\"A partir de qual rentabilidade a estratégia à vista passa a ganhar?\"*\n"
            "- *\"Se a valorização do imóvel for maior que X%, o consórcio vira o jogo?\"*\n"
            "- *\"Qual o maior lance que ainda vale a pena dar?\"*\n\n"
            "A **linha verde tracejada** marca o **ponto de cruzamento** — o valor exato do parâmetro "
            "em que as duas estratégias empatam. A **linha cinza pontilhada** marca o valor atual "
            "configurado na barra lateral, para você ver de que lado do cruzamento está."
        )

        # Essa aba compara especificamente Consórcio vs À Vista
        if cash_result is None or cons_result is None:
            st.info(
                "ℹ️ A análise de sensibilidade compara **Consórcio vs Compra à Vista**. "
                "Habilite as duas estratégias na barra lateral para usar esta aba."
            )
            st.stop()

        col_s1, col_s2 = st.columns([1, 2])

        with col_s1:
            sens_param = st.selectbox(
                "Parâmetro a variar",
                options=list(SENSITIVITY_PARAMS.keys()),
                format_func=lambda k: SENSITIVITY_PARAMS[k],
            )

            current_val = float(getattr(inp, sens_param))

            # Range inteligente por parâmetro
            range_defaults = {
                "investment_return_annual_net": (2.0, 20.0),
                "property_appreciation_annual": (0.0, 15.0),
                "bid_pct":                      (5.0, 80.0),
                "contemplation_year":           (1.0, float(inp.horizon_years)),
                "cash_purchase_year":           (1.0, float(inp.horizon_years)),
                "admin_fee_pct":                (5.0, 30.0),
                "adjustment_pre_annual":        (0.0, 15.0),
                "adjustment_post_annual":       (0.0, 15.0),
            }
            lo_def, hi_def = range_defaults.get(sens_param, (current_val * 0.5, current_val * 1.5))

            sens_min = st.number_input("Valor mínimo", value=round(lo_def, 1), step=0.5, format="%.1f")
            sens_max = st.number_input("Valor máximo", value=round(hi_def, 1), step=0.5, format="%.1f")
            n_steps = st.slider("Número de pontos", min_value=10, max_value=80, value=40)

            if sens_min >= sens_max:
                st.error("O valor mínimo deve ser menor que o máximo.")
                st.stop()

        sweep_values = list(np.linspace(sens_min, sens_max, n_steps))
        sweep_results = sensitivity_sweep(inp, sens_param, sweep_values)

        if not sweep_results:
            st.warning("Nenhum ponto válido na varredura. Ajuste o intervalo.")
        else:
            crossover = find_crossover(sweep_results)

            xs = [r["value"] for r in sweep_results]
            y_cash = [r["cash_total"] for r in sweep_results]
            y_cons = [r["cons_total"] for r in sweep_results]
            y_diff = [r["diff"] for r in sweep_results]

            param_label = SENSITIVITY_PARAMS[sens_param]

            with col_s2:
                # --- Métricas do cruzamento ---
                if crossover is not None:
                    below_winner = "Consórcio" if sweep_results[0]["diff"] > 0 else label_cash
                    above_winner = label_cash if sweep_results[0]["diff"] > 0 else "Consórcio"
                    st.success(
                        f"⚖️ **Ponto de cruzamento: {crossover:.2f}** em '{param_label}' "
                        f"— abaixo disso o **{below_winner}** vence; "
                        f"acima, o **{above_winner}** vence."
                    )
                else:
                    winner_sens = "Consórcio" if sweep_results[-1]["diff"] > 0 else label_cash
                    st.info(f"ℹ️ Não há cruzamento no intervalo analisado. **{winner_sens}** vence em todos os pontos.")

            # --- Gráfico 1: Patrimônio final vs parâmetro ---
            fig_s1 = go.Figure()
            fig_s1.add_trace(go.Scatter(
                x=xs, y=y_cash, name="À Vista — Total",
                mode="lines", line=dict(color="#1f77b4", width=2.5),
                hovertemplate=f"{param_label}: %{{x:.2f}}<br>À Vista: R$ %{{y:,.0f}}<extra></extra>"
            ))
            fig_s1.add_trace(go.Scatter(
                x=xs, y=y_cons, name="Consórcio — Total",
                mode="lines", line=dict(color="#ff7f0e", width=2.5),
                hovertemplate=f"{param_label}: %{{x:.2f}}<br>Consórcio: R$ %{{y:,.0f}}<extra></extra>"
            ))

            # Linha do valor atual
            fig_s1.add_vline(
                x=current_val, line_dash="dot", line_color="gray",
                annotation_text=f"Atual: {current_val:.2f}",
                annotation_position="top right", annotation_font_size=11
            )
            # Linha do cruzamento
            if crossover is not None:
                fig_s1.add_vline(
                    x=crossover, line_dash="dash", line_color="#2ca02c", line_width=2,
                    annotation_text=f"Cruzamento: {crossover:.2f}",
                    annotation_position="top left",
                    annotation_font=dict(color="#2ca02c", size=12)
                )

            fig_s1.update_layout(
                title=f"Patrimônio Total Final vs {param_label}",
                xaxis_title=param_label,
                yaxis_title="Patrimônio Total Final (R$)",
                yaxis=dict(tickformat=",.0f", tickprefix="R$ "),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                height=400,
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig_s1, use_container_width=True)

            # --- Gráfico 2: Diferença vs parâmetro ---
            colors_s = ["#ff7f0e" if d > 0 else "#1f77b4" for d in y_diff]
            fig_s2 = go.Figure()
            fig_s2.add_trace(go.Bar(
                x=xs, y=y_diff,
                marker_color=colors_s,
                name="Diferença (Consórcio − À Vista)",
                hovertemplate=f"{param_label}: %{{x:.2f}}<br>Diferença: R$ %{{y:,.0f}}<extra></extra>"
            ))
            fig_s2.add_hline(y=0, line_color="black", line_width=1)

            if crossover is not None:
                fig_s2.add_vline(
                    x=crossover, line_dash="dash", line_color="#2ca02c", line_width=2,
                    annotation_text=f"Cruzamento: {crossover:.2f}",
                    annotation_position="top left",
                    annotation_font=dict(color="#2ca02c", size=12)
                )
            fig_s2.add_vline(
                x=current_val, line_dash="dot", line_color="gray",
                annotation_text=f"Atual: {current_val:.2f}",
                annotation_position="top right", annotation_font_size=11
            )

            fig_s2.update_layout(
                title=f"Vantagem do Consórcio vs {param_label}",
                xaxis_title=param_label,
                yaxis_title="Diferença (R$) — positivo = consórcio vence",
                yaxis=dict(tickformat=",.0f", tickprefix="R$ "),
                height=320,
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_s2, use_container_width=True)

            # --- Tabela resumida ---
            with st.expander("Ver tabela de dados da varredura"):
                df_sweep = pd.DataFrame({
                    param_label: [f"{r['value']:.2f}" for r in sweep_results],
                    "À Vista — Total": [fmt_brl(r["cash_total"]) for r in sweep_results],
                    "Consórcio — Total": [fmt_brl(r["cons_total"]) for r in sweep_results],
                    "Diferença (Cons − AV)": [fmt_brl(r["diff"]) for r in sweep_results],
                    "Vencedor": ["Consórcio 🟠" if r["diff"] > 0 else "À Vista 🔵" for r in sweep_results],
                })
                st.dataframe(df_sweep, hide_index=True, use_container_width=True)

except Exception as e:
    st.error(f"Erro na simulação: {e}")
    st.exception(e)
