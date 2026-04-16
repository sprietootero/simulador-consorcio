import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, replace
from typing import Optional
import numpy as np

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
    monthly_property_cost_after_purchase: float
    monthly_precision_rounding: bool
    monthly_savings: float
    savings_inflation_annual: float
    ir_aliquota_pct: float
    investment_return_annual_gross: float
    no_cash_purchase: bool  # Se True, estratégia "à vista" nunca compra — só deixa rendendo

def missing_required_fields(inp: Inputs) -> list[str]:
    """Retorna lista de campos obrigatórios ainda não preenchidos (valor 0)."""
    missing = []
    if inp.initial_cash <= 0:
        missing.append("Caixa inicial")
    if inp.property_price_today <= 0:
        missing.append("Preço do imóvel hoje")
    if inp.consortium_credit_today <= 0:
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

            if purchased:
                rent_saved = rental_value_at_month(inp.monthly_rent_saved_after_purchase, inp.rent_inflation_annual, m)
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
                rent_saved = rental_value_at_month(
                    inp.monthly_rent_saved_after_purchase, inp.rent_inflation_annual, m
                )
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
</style>
""", unsafe_allow_html=True)

st.title("🏡 Simulador Financeiro: Consórcio vs Compra à Vista")

st.markdown("""
Este simulador compara, mês a mês, duas formas de obter um imóvel:

- **💵 Compra à Vista** — você mantém o dinheiro investido rendendo e saca no ano escolhido para comprar o imóvel à vista.
- **🏦 Consórcio** — você paga parcelas mensais, é contemplado em um determinado ano (por lance), recebe a carta de crédito e adquire o imóvel. O restante do dinheiro segue rendendo no caixa.

Ao final do horizonte, o simulador compara o **patrimônio total** (caixa investido + valor do imóvel) das duas estratégias. Ajuste os parâmetros na barra lateral à esquerda — cada campo tem uma explicação 🛈 ao passar o mouse.

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

with st.sidebar.expander("💵 Estratégia: À Vista"):
    st.caption(
        "Nesta estratégia, seu caixa fica **rendendo até o ano da compra**. "
        "No ano escolhido, você saca o valor necessário e compra o imóvel à vista. "
        "O restante continua rendendo até o fim do horizonte."
    )
    no_cash_purchase = st.checkbox(
        "Não comprar — só deixar rendendo",
        value=False,
        help="Marque para comparar o consórcio contra a estratégia de simplesmente manter o capital investido, sem comprar imóvel algum. Útil para entender se vale a pena se imobilizar no imóvel."
    )
    if not no_cash_purchase:
        cash_purchase_year = st.number_input(
            "Ano da compra à vista", value=1.0, min_value=0.1, step=0.5, format="%.1f",
            help="Em que ano do horizonte você sacaria o dinheiro para comprar o imóvel à vista. "
                 "Ano 1 = compra imediata (após 12 meses). Quanto maior, mais tempo o caixa rende antes da compra, "
                 "mas o imóvel também fica mais caro pela valorização."
        )
        cash_purchase_extra_cost_pct = st.number_input(
            "Custos extras na compra (%)", value=0.0, min_value=0.0, step=0.1, format="%.2f",
            help="Percentual adicional sobre o preço do imóvel para cobrir ITBI, escritura, registro em cartório, etc. "
                 "Referência típica: 4% a 6% (ITBI ~3% + escritura/registro ~1-2%)."
        )
    else:
        cash_purchase_year = float(horizon_years) + 999  # nunca compra
        cash_purchase_extra_cost_pct = 0.0

with st.sidebar.expander("🏦 Estratégia: Consórcio"):
    st.caption(
        "No consórcio, você paga **parcelas mensais** e é contemplado em um determinado mês "
        "(aqui, assumimos contemplação **por lance**). Até a contemplação, pode-se pagar uma fração "
        "da parcela cheia. Após o lance, recebe a carta de crédito e compra o imóvel."
    )
    consortium_credit_today = st.number_input(
        "Carta de crédito (R$)", value=0.0, step=50_000.0, min_value=0.0, format="%.2f",
        help="Valor da carta de crédito do consórcio em reais de hoje. "
             "Normalmente igual ou próximo ao preço do imóvel que você quer comprar. "
             "A carta é reajustada anualmente pelo índice pré-contemplação (geralmente INCC)."
    )
    term_months = st.number_input(
        "Prazo total do plano (meses)", value=180, min_value=12, max_value=360, step=12,
        help="Duração total do consórcio em meses. Valores comuns: 120, 180, 200, 240. "
             "As parcelas pré-contemplação são proporcionais (carta + taxas ÷ prazo). "
             "Após a contemplação, o saldo devedor é diluído nos meses restantes do plano."
    )
    contemplation_year = st.number_input(
        "Ano de contemplação projetado", value=5.0, min_value=0.1, step=0.5, format="%.1f",
        help="Em que ano você espera ser contemplado (dar o lance vencedor). "
             "Este é o campo mais especulativo — depende do grupo, do valor do lance e da sorte/estratégia. "
             "Teste cenários de contemplação em anos diferentes usando a aba de Sensibilidade."
    )
    pre_contemplation_payment_fraction = st.number_input(
        "Fração da parcela antes da contemplação", value=0.5, min_value=0.01, max_value=1.0, step=0.05, format="%.2f",
        help="Quanto da parcela cheia você paga ANTES de ser contemplado. "
             "Ex: 0,5 = paga metade da parcela cheia. Alguns consórcios permitem parcela reduzida enquanto se aguarda contemplação. "
             "Use 1,0 se for pagar a parcela cheia desde o início."
    )
    bid_pct = st.number_input(
        "Lance total (% da carta)", value=40.0, min_value=0.0, max_value=100.0, step=1.0, format="%.1f",
        help="Percentual do valor do plano oferecido como lance para ser contemplado. "
             "Lances maiores aumentam a chance de contemplação mais cedo, mas exigem mais capital imediato. "
             "No Brasil, lances vencedores costumam ficar entre 30% e 60%, dependendo do grupo."
    )
    embedded_bid_pct_of_bid = st.number_input(
        "Lance embutido (% do lance total)", value=0.0, min_value=0.0, max_value=100.0, step=5.0, format="%.1f",
        help="Parte do lance que é descontada da própria carta de crédito (não sai do seu bolso), "
             "mas em troca você recebe uma carta menor. Ex: carta de R$ 500k, lance total de 30% (R$ 150k), "
             "embutido 50% → R$ 75k sai do bolso, R$ 75k reduz a carta (que vira R$ 425k). "
             "Nem toda administradora permite lance embutido — verifique as regras do seu plano."
    )
    bid_base = st.selectbox(
        "Base do lance", ["adjusted_credit", "original_credit"],
        format_func=lambda x: "Sobre carta reajustada" if x == "adjusted_credit" else "Sobre carta original",
        help="Sobre qual valor o percentual do lance é calculado: a carta já corrigida pelo INCC até o mês da contemplação "
             "(mais comum) ou a carta original de hoje. A maior parte das administradoras usa a carta reajustada."
    )
    reduce_installment_after_bid = st.checkbox(
        "Lance abate saldo devedor futuro", value=True,
        help="Se marcado, o lance reduz o saldo a pagar após a contemplação (diminui as parcelas futuras). "
             "Se desmarcado, o lance apenas serve para contemplar e não reduz o valor total a pagar pelo plano. "
             "A regra depende da administradora — verifique seu contrato."
    )
    admin_fee_pct = st.number_input(
        "Taxa administrativa total (%)", value=20.0, min_value=0.0, step=0.5, format="%.2f",
        help="Taxa da administradora, cobrada sobre o valor da carta e diluída nas parcelas. "
             "No Brasil, costuma ficar entre 15% e 25% do valor da carta. É o principal custo do consórcio."
    )
    reserve_fund_pct = st.number_input(
        "Fundo de reserva (%)", value=1.0, min_value=0.0, step=0.5, format="%.2f",
        help="Fundo comum do grupo para cobrir inadimplência e sorteios extras. "
             "Valor típico: 1% a 3% do valor da carta. Também é diluído nas parcelas."
    )
    adjustment_pre_annual = st.number_input(
        "Reajuste anual pré-contemplação (INCC) (%)", value=5.7, step=0.1, format="%.2f",
        help="Reajuste anual aplicado à carta de crédito e às parcelas ANTES da contemplação. "
             "Geralmente é o INCC (Índice Nacional de Custo da Construção). Histórico recente: entre 4% e 10% a.a."
    )
    adjustment_post_annual = st.number_input(
        "Reajuste anual pós-contemplação (IPCA) (%)", value=5.0, step=0.1, format="%.2f",
        help="Reajuste anual aplicado ao saldo devedor e às parcelas APÓS a contemplação. "
             "Geralmente é o IPCA ou INCC — depende do contrato. Histórico recente do IPCA: 3% a 10% a.a."
    )
    insurance_monthly_pct_of_credit = st.number_input(
        "Seguro mensal (% da carta)", value=0.035, min_value=0.0, step=0.01, format="%.3f",
        help="Seguro de vida/prestamista cobrado mensalmente como percentual da carta de crédito vigente. "
             "Valor típico: 0,03% a 0,05% ao mês. Ex: 0,035% sobre carta de R$ 500 mil = R$ 175/mês. "
             "É um custo adicional além da parcela de amortização."
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

with st.sidebar.expander("🏠 Pós-compra (ambas as estratégias)"):
    st.caption(
        "Efeitos financeiros que passam a valer **após a posse do imóvel** — em ambas as estratégias, "
        "a partir do mês da compra (à vista) ou da contemplação (consórcio)."
    )
    monthly_rent_saved_after_purchase = st.number_input(
        "Aluguel evitado após compra (R$/mês)", value=0.0, min_value=0.0, step=500.0, format="%.2f",
        help="Quanto você deixa de pagar de aluguel por mês ao ser dono do imóvel. "
             "Entra como entrada mensal no caixa após a compra. Se não morava de aluguel, deixe em 0."
    )
    rent_inflation_annual = st.number_input(
        "Inflação do aluguel a.a. (%)", value=5.0, step=0.1, format="%.2f",
        help="Reajuste anual do aluguel evitado. Geralmente IGPM ou IPCA. Histórico: 3% a 10% a.a."
    )
    monthly_property_cost_after_purchase = st.number_input(
        "Custo mensal do imóvel após compra (R$)", value=0.0, min_value=0.0, step=100.0, format="%.2f",
        help="Custos fixos de ter o imóvel: IPTU mensalizado, condomínio, manutenção, seguro residencial, etc. "
             "Entra como saída mensal do caixa após a compra."
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
        monthly_property_cost_after_purchase=monthly_property_cost_after_purchase,
        monthly_precision_rounding=monthly_precision_rounding,
        monthly_savings=monthly_savings,
        savings_inflation_annual=savings_inflation_annual,
        ir_aliquota_pct=ir_aliquota_pct,
        investment_return_annual_gross=investment_return_annual_gross,
        no_cash_purchase=no_cash_purchase,
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

    cash_result = simulate_cash_later(inp)
    cons_result = simulate_consortium(inp)

    # Alertas de saldo negativo
    if cash_result["went_negative"]:
        st.warning("⚠️ **Compra à Vista:** o saldo financeiro ficou negativo em algum momento da simulação. "
                   "Considere aumentar o caixa inicial ou o aporte mensal.")
    if cons_result["went_negative"]:
        st.warning("⚠️ **Consórcio:** o saldo financeiro ficou negativo em algum momento da simulação. "
                   "Considere aumentar o caixa inicial ou o aporte mensal.")

    label_cash = "Só Rendendo" if inp.no_cash_purchase else "Compra à Vista"

    diff = cons_result["final_total_wealth"] - cash_result["final_total_wealth"]
    winner_label = f"Consórcio" if diff > 0 else label_cash
    winner_class = "winner-consortium" if diff > 0 else "winner-cash"

    st.markdown(
        f'<div class="winner-box {winner_class}">'
        f'🏆 Estratégia vencedora: <strong>{winner_label}</strong> '
        f'— vantagem de <strong>{fmt_brl(abs(diff))}</strong> ao final do período'
        f'</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Gráfico", "📝 Resumo", "🔎 Auditoria", "📅 Ano a Ano", "📉 Sensibilidade"])

    # ---- GRÁFICO ----
    with tab1:
        st.markdown(
            "### Evolução do Patrimônio ao Longo do Tempo\n"
            "As **linhas sólidas** mostram o patrimônio total (caixa + valor do imóvel). "
            "As **linhas pontilhadas** mostram apenas o caixa financeiro. "
            "As **linhas verticais tracejadas** marcam os eventos-chave: compra à vista 🔵 e contemplação 🟠.\n\n"
            "🟦 Azul = À Vista | 🟧 Laranja = Consórcio"
        )
        df_cash = pd.DataFrame(cash_result["records"])
        df_cons = pd.DataFrame(cons_result["records"])

        fig = go.Figure()

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

        if not inp.no_cash_purchase:
            fig.add_vline(
                x=inp.cash_purchase_year, line_dash="dash", line_color="#1f77b4",
                annotation_text=f"Compra à Vista (Ano {inp.cash_purchase_year:.1f})",
                annotation_position="top left", annotation_font_size=11
            )
        fig.add_vline(
            x=inp.contemplation_year, line_dash="dash", line_color="#ff7f0e",
            annotation_text=f"Contemplação (Ano {inp.contemplation_year:.1f})",
            annotation_position="top right", annotation_font_size=11
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

        st.markdown(
            "#### Diferença de Patrimônio\n"
            "Quando a barra está **laranja (positiva)**, o consórcio está à frente naquele ano. "
            "Quando está **azul (negativa)**, a estratégia à vista está à frente. "
            "Observe em que momento acontece o cruzamento — esse costuma ser o ponto decisivo da análise."
        )

        # Gráfico de diferença
        diff_series = [
            c["total_wealth"] - a["total_wealth"]
            for c, a in zip(cons_result["records"], cash_result["records"])
        ]
        years_series = [r["year"] for r in cash_result["records"]]
        colors_diff = ["#ff7f0e" if d > 0 else "#1f77b4" for d in diff_series]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=years_series, y=diff_series,
            marker_color=colors_diff,
            name="Diferença (Consórcio − À Vista)",
            hovertemplate="Ano %{x:.1f}<br>Diferença: R$ %{y:,.0f}<extra></extra>"
        ))
        fig2.add_hline(y=0, line_color="gray", line_width=1)
        fig2.update_layout(
            title=f"Diferença de Patrimônio Total (Consórcio − {label_cash})",
            xaxis_title="Ano",
            yaxis_title="Diferença (R$)",
            yaxis=dict(tickformat=",.0f", tickprefix="R$ "),
            height=300,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---- RESUMO ----
    with tab2:
        st.markdown(
            "### Posição Final no Horizonte\n"
            "Compare os resultados das duas estratégias ao fim do período. **Patrimônio Total** é o que "
            "você teria se vendesse tudo no último dia: caixa investido + valor de mercado do imóvel. "
            "**Desembolso Total do Bolso** é quanto saiu efetivamente do seu caixa ao longo da simulação."
        )
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 💵 {label_cash}")
            st.metric("Patrimônio Total Final", fmt_brl(cash_result["final_total_wealth"]))
            st.metric("Patrimônio Financeiro (Caixa)", fmt_brl(cash_result["final_financial_wealth"]))
            st.metric("Patrimônio Imobiliário", fmt_brl(cash_result["final_real_estate_wealth"]))
            st.metric("Desembolso Total do Bolso", fmt_brl(cash_result["out_of_pocket_total"]))
            st.metric("IR pago sobre rendimentos", fmt_brl(cash_result["cumulative_ir_paid"]))
            if inp.monthly_savings > 0:
                st.metric("Total aportado no período", fmt_brl(cash_result["cumulative_savings"]))

        with col2:
            st.markdown("### 🏦 Consórcio")
            st.metric("Patrimônio Total Final", fmt_brl(cons_result["final_total_wealth"]))
            st.metric("Patrimônio Financeiro (Caixa)", fmt_brl(cons_result["final_financial_wealth"]))
            st.metric("Patrimônio Imobiliário", fmt_brl(cons_result["final_real_estate_wealth"]))
            st.metric("Desembolso Total do Bolso", fmt_brl(cons_result["out_of_pocket_total"]))
            if cons_result["final_cumulative_property_topup"] > 0:
                st.metric(
                    "Complemento próprio p/ cobrir imóvel",
                    fmt_brl(cons_result["final_cumulative_property_topup"]),
                    help="Valor tirado do caixa porque a carta líquida (após lance embutido) não cobriu o preço do imóvel"
                )
            if cons_result["final_cumulative_bid_own_cash"] > 0:
                st.metric("Lance do próprio bolso", fmt_brl(cons_result["final_cumulative_bid_own_cash"]))
            if cons_result["final_cumulative_bid_embedded"] > 0:
                st.metric("Lance embutido (da carta)", fmt_brl(cons_result["final_cumulative_bid_embedded"]))
            st.metric("IR pago sobre rendimentos", fmt_brl(cons_result["cumulative_ir_paid"]))
            if inp.monthly_savings > 0:
                st.metric("Total aportado no período", fmt_brl(cons_result["cumulative_savings"]))

        st.divider()
        st.markdown("### 📊 Comparativo Final")
        comp_metrics = ["Patrimônio Total", "Patrimônio Financeiro", "Patrimônio Imobiliário", "Desembolso Total", "IR pago (rendimentos)"]
        comp_cash = [
            cash_result["final_total_wealth"],
            cash_result["final_financial_wealth"],
            cash_result["final_real_estate_wealth"],
            cash_result["out_of_pocket_total"],
            cash_result["cumulative_ir_paid"],
        ]
        comp_cons = [
            cons_result["final_total_wealth"],
            cons_result["final_financial_wealth"],
            cons_result["final_real_estate_wealth"],
            cons_result["out_of_pocket_total"],
            cons_result["cumulative_ir_paid"],
        ]

        # Adicionar linhas detalhadas do consórcio se relevantes
        if cons_result["final_cumulative_property_topup"] > 0:
            comp_metrics.append("Complemento próprio p/ imóvel")
            comp_cash.append(0.0)
            comp_cons.append(cons_result["final_cumulative_property_topup"])
        if cons_result["final_cumulative_bid_own_cash"] > 0:
            comp_metrics.append("Lance do próprio bolso")
            comp_cash.append(0.0)
            comp_cons.append(cons_result["final_cumulative_bid_own_cash"])
        if cons_result["final_cumulative_bid_embedded"] > 0:
            comp_metrics.append("Lance embutido (da carta)")
            comp_cash.append(0.0)
            comp_cons.append(cons_result["final_cumulative_bid_embedded"])

        comp_df = pd.DataFrame({
            "Métrica": comp_metrics,
            "À Vista": [fmt_brl(v) for v in comp_cash],
            "Consórcio": [fmt_brl(v) for v in comp_cons],
            "Diferença (Consórcio − À Vista)": [fmt_brl(c - a) for c, a in zip(comp_cons, comp_cash)],
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ---- AUDITORIA ----
    with tab3:
        st.markdown("### 🔎 Dados de Auditoria do Consórcio")
        st.markdown(
            "Esta aba detalha como os valores do consórcio foram calculados, para você conferir a mecânica:\n\n"
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

        st.divider()
        st.markdown("**Decomposição do desembolso total no consórcio**")
        decomp = {
            "Parcelas": cons_result["final_cumulative_installments"],
            "Seguro": cons_result["final_cumulative_insurance"],
            "Lance (bolso)": cons_result["final_cumulative_bid_own_cash"],
            "Complemento imóvel": cons_result["final_cumulative_property_topup"],
        }
        fig3 = px.pie(
            names=list(decomp.keys()),
            values=list(decomp.values()),
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig3.update_traces(textinfo="label+percent", hovertemplate="%{label}: R$ %{value:,.0f}<extra></extra>")
        fig3.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    # ---- ANO A ANO ----
    with tab4:
        st.markdown(
            "### 📅 Evolução Anual do Patrimônio\n"
            "Tabela com a foto do patrimônio ao final de cada ano, nas duas estratégias. "
            "Útil para identificar o **ano do cruzamento** (quando a diferença muda de sinal) e para "
            "exportar os dados para análise externa. A coluna **Parcela Média** do consórcio é a média "
            "dos 12 meses do ano — antes da contemplação tende a ser menor (fração da parcela cheia), "
            "e depois sobe até o fim do plano."
        )

        # Filtra apenas anos inteiros
        cash_annual = [r for r in cash_result["records"] if r["month"] % 12 == 0]
        cons_annual = [r for r in cons_result["records"] if r["month"] % 12 == 0]

        # Parcela média do consórcio no ano: média dos 12 meses anteriores
        def avg_installment_for_year(records, year_idx):
            """Média da parcela mensal nos 12 meses do ano."""
            month_end = year_idx * 12
            month_start = month_end - 11
            if month_start < 0:
                return 0.0
            monthly = [r["monthly_outflow"] for r in records if month_start <= r["month"] <= month_end]
            return sum(monthly) / len(monthly) if monthly else 0.0

        df_years = pd.DataFrame({
            "Ano": [int(r["year"]) for r in cash_annual],
            "À Vista: Caixa": [fmt_brl(r["financial_wealth"]) for r in cash_annual],
            "À Vista: Imóvel": [fmt_brl(r["real_estate_wealth"]) for r in cash_annual],
            "À Vista: Total": [fmt_brl(r["total_wealth"]) for r in cash_annual],
            "Consórcio: Parcela Média": [
                fmt_brl(avg_installment_for_year(cons_result["records"], int(r["year"])))
                for r in cons_annual
            ],
            "Consórcio: Caixa": [fmt_brl(r["financial_wealth"]) for r in cons_annual],
            "Consórcio: Imóvel": [fmt_brl(r["real_estate_wealth"]) for r in cons_annual],
            "Consórcio: Total": [fmt_brl(r["total_wealth"]) for r in cons_annual],
            "Diferença (Cons − AV)": [
                fmt_brl(c["total_wealth"] - a["total_wealth"])
                for c, a in zip(cons_annual, cash_annual)
            ],
        })

        st.dataframe(df_years, hide_index=True, use_container_width=True)

        csv = df_years.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar tabela como CSV",
            data=csv,
            file_name="simulacao_consorcio_vs_avista.csv",
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
