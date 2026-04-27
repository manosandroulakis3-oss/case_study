"""
team.blue Cohort Analysis — Streamlit Dashboard
Built to mirror the strategic report. BI-tool styling.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from data_prep import (
    load_all_data, apply_filters, compute_kpis,
    compute_kpi_table, compute_retention_curve, compute_logo_retention_curve
)
from charts import (
    chart_mrr_over_time, chart_arpu_over_time, chart_nrr_over_time,
    chart_mrr_movement, chart_retention_curve, chart_logo_retention_curve,
    chart_mrr_by_dimension, chart_renewal_sankey,
)

# =============================================================================
# Page config & styling
# =============================================================================
st.set_page_config(
    page_title="team.blue | Cohort Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = Path(__file__).parent / "style.css"
with open(CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =============================================================================
# Helpers
# =============================================================================
def fmt_currency(v, prefix='€'):
    if v is None or pd.isna(v):
        return '—'
    if abs(v) >= 1_000_000:
        return f'{prefix}{v/1_000_000:.1f}M'
    if abs(v) >= 1_000:
        return f'{prefix}{v/1_000:.0f}K'
    if abs(v) < 100:
        return f'{prefix}{v:.2f}'  # ARPU and similar small values keep cents
    return f'{prefix}{v:,.0f}'


def fmt_pct(v):
    return '—' if pd.isna(v) else f'{v:.1f}%'


def kpi_card(label, value, delta=None, delta_dir=None):
    delta_html = ''
    if delta is not None:
        cls = {
            'positive': 'kpi-delta-positive',
            'negative': 'kpi-delta-negative',
        }.get(delta_dir, 'kpi-delta-neutral')
        arrow = '▲' if delta_dir == 'positive' else ('▼' if delta_dir == 'negative' else '●')
        delta_html = f"<div class='kpi-delta {cls}'>{arrow} {delta}</div>"

    st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def render_logo_retention_table(mrr_subset, customers_subset, title=None):
    """Render the colour-coded customer retention milestone table.
    Reusable across tabs — call with a filtered mrr/customers slice."""
    st.markdown(
        f"<div class='section-header'>{title or 'Customer Retention by Acquisition Cohort'}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='caption'>% of cohort customers still active at key contract milestones. "
        "Drops between M12→M13 reveal first-year renewal rate; M24→M25 the second. "
        "Cells marked — are right-censored (cohort hasn't had enough observation time).</div>",
        unsafe_allow_html=True
    )

    logo_ret = compute_logo_retention_curve(mrr_subset, customers_subset)
    if len(logo_ret) == 0:
        st.info("Not enough data for retention table.")
        return

    key_months = [6, 12, 13, 24, 25, 36]
    logo_ret = logo_ret[logo_ret['cohort_year'] < 2026].copy()
    logo_ret.loc[logo_ret['mature_pct'] < 80, 'pct_retained'] = None

    logo_table = (logo_ret[logo_ret['months_since'].isin(key_months)]
                  .pivot(index='cohort_year', columns='months_since',
                         values='pct_retained')
                  .round(1))
    for m in key_months:
        if m not in logo_table.columns:
            logo_table[m] = None
    logo_table = logo_table[key_months]
    logo_table.columns = [f'Month {m}' for m in logo_table.columns]
    logo_table.index = [f'Cohort {int(c)}' for c in logo_table.index]
    logo_table.index.name = ''

    def _color(val):
        if pd.isna(val):
            return 'background-color: #f3f4f6; color: #9ca3af;'
        if val >= 95:
            return 'background-color: #d1fae5; color: #064e3b;'
        if val >= 75:
            return 'background-color: #ecfccb; color: #1a2e05;'
        if val >= 50:
            return 'background-color: #fef9c3; color: #713f12;'
        if val >= 25:
            return 'background-color: #fed7aa; color: #7c2d12;'
        return 'background-color: #fecaca; color: #7f1d1d;'

    styled = (logo_table.style
              .format('{:.1f}%', na_rep='—')
              .map(_color))
    st.dataframe(styled, use_container_width=True)


# =============================================================================
# Load data
# =============================================================================
with st.spinner('Loading data…'):
    customers, invoices, mrr = load_all_data()


# =============================================================================
# Sidebar — slicers
# =============================================================================
st.sidebar.markdown("### 🔍 Filters")
st.sidebar.caption("All charts and KPIs respond to these.")

date_range = None  # Date range filter removed — full data range is always used

_country_options = [c for c in sorted(customers['country'].unique()) if c != 'Estonia']
countries = st.sidebar.multiselect(
    "Country", _country_options,
    default=None, placeholder="All countries",
)

cohorts = st.sidebar.multiselect(
    "Acquisition cohort", sorted(customers['cohort_year'].unique()),
    default=None, placeholder="All cohorts",
)

product_groups = st.sidebar.multiselect(
    "Product group", sorted(invoices['product_group'].unique()),
    default=None, placeholder="All groups",
)

product_subgroups = st.sidebar.multiselect(
    "Product subgroup", sorted(invoices['product_subgroup'].unique()),
    default=None, placeholder="All subgroups",
)

_sub_order = ['1 months', '12 months', '24 months', '60 months', '120 months', 'Other']
_sub_available = [s for s in _sub_order if s in customers['subscription_type'].unique()]
contract_lengths = st.sidebar.multiselect(
    "Contract length",
    _sub_available,
    default=None, placeholder="All lengths",
)

platform_depths = st.sidebar.multiselect(
    "Platform depth", sorted(customers['platform_depth'].unique()),
    default=None, placeholder="All depths",
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='caption'>Strategic Report · April 2026<br>Manos Androulakis</div>",
                    unsafe_allow_html=True)


# =============================================================================
# Apply filters
# =============================================================================
cust_f, inv_f, mrr_f = apply_filters(
    customers, invoices, mrr,
    date_range=date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None,
    countries=countries or None,
    cohorts=cohorts or None,
    product_groups=product_groups or None,
    product_subgroups=product_subgroups or None,
    contract_lengths=contract_lengths or None,
    platform_depths=platform_depths or None,
)

kpis = compute_kpis(cust_f, inv_f, mrr_f)

# Time-series charts visually start at 2021 (matches the report).
# KPI tables and strip use the full mrr_f so numbers match the notebook's
# all-data calculations.
mrr_chart = mrr_f[mrr_f['mrr_month'] >= pd.Timestamp('2021-01-01')]


# =============================================================================
# Header
# =============================================================================
st.markdown("<h1 class='dashboard-title'>team.blue · Cohort Analysis</h1>",
            unsafe_allow_html=True)
st.markdown(f"<div class='dashboard-subtitle'>"
            f"{len(customers):,} customers · 12 European markets · Jan 2021 – Apr 2026"
            f"</div>", unsafe_allow_html=True)


# =============================================================================
# KPI strip
# =============================================================================
cols = st.columns(6)
with cols[0]:
    kpi_card("Customers", f"{kpis['customers']:,}", "19.4% CAGR decline", "negative")
with cols[1]:
    kpi_card("Avg MRR / month", fmt_currency(kpis['avg_mrr']), "35.5% CAGR", "positive")
with cols[2]:
    kpi_card("ARPU", fmt_currency(kpis['arpu']) + "/mo", "4.9% CAGR", "positive")
with cols[3]:
    kpi_card("Renewal Rate", fmt_pct(kpis['renewal_rate']), "21.1% CAGR decline", "negative")
with cols[4]:
    kpi_card("Avg LTV", fmt_currency(kpis['ltv']), "Full observation", "neutral")
with cols[5]:
    kpi_card("NRR at Expiry", fmt_pct(kpis['nrr_expiry']),
             "Above 100% = growth", "neutral")

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# Tabs
# =============================================================================
tab_state, tab_depth, tab_country, tab_product, tab_contract, tab_action = st.tabs([
    "📈 State of Business",
    "🔢 Platform Depth",
    "🌍 Country",
    "📦 Product",
    "📅 Contract Length",
    "🎯 Action Plan",
])


# -----------------------------------------------------------------------------
# Tab 1 — State of the Business
# -----------------------------------------------------------------------------
with tab_state:
    # Headline chart full-width
    st.plotly_chart(chart_mrr_over_time(mrr_chart), use_container_width=True)

    # ARPU + NRR side by side
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_arpu_over_time(mrr_chart), use_container_width=True)
    with c2:
        st.plotly_chart(chart_nrr_over_time(mrr_chart), use_container_width=True)

    # MRR movement gets its own row — bars need horizontal space
    st.plotly_chart(chart_mrr_movement(mrr_chart), use_container_width=True)

    # Cohort retention — table first (milestones), then curve (full shape)
    render_logo_retention_table(mrr_chart, customers)

    # Revenue retention curve — full shape, includes the M12 expansion spike
    ret = compute_retention_curve(mrr_chart, customers)
    st.plotly_chart(chart_retention_curve(ret), use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 2 — Platform Depth
# -----------------------------------------------------------------------------
with tab_depth:
    st.markdown("<div class='section-header'>KPIs by Platform Depth</div>",
                unsafe_allow_html=True)
    table = compute_kpi_table(cust_f, inv_f, mrr_f, 'platform_depth')
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.plotly_chart(chart_mrr_by_dimension(mrr_chart, 'platform_depth',
                    'MRR by Platform Depth'), use_container_width=True)

    # Retention table covering all customers in the current filter
    render_logo_retention_table(mrr_chart, customers)


# -----------------------------------------------------------------------------
# Tab 3 — Country
# -----------------------------------------------------------------------------
with tab_country:
    # Exclude Estonia (only 18 customers — sample too small per the report's appendix).
    # Headline KPIs above keep all customers; only country-level views drop Estonia.
    cust_c = cust_f[cust_f['country'] != 'Estonia']
    inv_c = inv_f[inv_f['country'] != 'Estonia']
    mrr_c = mrr_f[mrr_f['country'] != 'Estonia']
    mrr_chart_c = mrr_chart[mrr_chart['country'] != 'Estonia']

    st.markdown("<div class='section-header'>KPIs by Country</div>",
                unsafe_allow_html=True)
    table = compute_kpi_table(cust_c, inv_c, mrr_c, 'country')
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.plotly_chart(chart_mrr_by_dimension(mrr_chart_c, 'country', 'MRR by Country'),
                    use_container_width=True)

    # Retention table — for selected countries (Estonia excluded)
    render_logo_retention_table(mrr_chart_c, customers)

    # Single cohort retention chart — respects whatever's selected in the country filter
    selected_countries = sorted([c for c in mrr_chart_c['country'].unique() if c == c])  # drop NaN
    if len(selected_countries) == 1:
        retention_title = f'Cohort Retention — {selected_countries[0]}'
    elif len(selected_countries) <= 3:
        retention_title = f'Cohort Retention — {", ".join(map(str, selected_countries))}'
    else:
        retention_title = f'Cohort Retention — {len(selected_countries)} countries selected'

    ret = compute_retention_curve(mrr_chart_c, customers)
    st.plotly_chart(chart_retention_curve(ret, retention_title), use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 4 — Product
# -----------------------------------------------------------------------------
with tab_product:
    st.markdown("<div class='section-header'>KPIs by Product Group</div>",
                unsafe_allow_html=True)
    table = compute_kpi_table(cust_f, inv_f, mrr_f, 'product_group')
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.plotly_chart(chart_mrr_by_dimension(mrr_chart, 'product_group', 'MRR by Product Group'),
                    use_container_width=True)

    # Retention table for current filter
    render_logo_retention_table(mrr_chart, customers)


# -----------------------------------------------------------------------------
# Tab 5 — Contract Length
# -----------------------------------------------------------------------------
with tab_contract:
    st.markdown("<div class='section-header'>KPIs by Contract Length</div>",
                unsafe_allow_html=True)
    table = compute_kpi_table(cust_f, inv_f, mrr_f, 'subscription_type')
    st.dataframe(table, use_container_width=True, hide_index=True)

    # MRR by contract length full-width
    st.plotly_chart(
        chart_mrr_by_dimension(mrr_chart, 'subscription_type', 'MRR by Contract Length'),
        use_container_width=True
    )

    # Retention table — for whatever's selected in the filter
    render_logo_retention_table(mrr_chart, customers)

    # Sankey on its own row — needs full width
    st.markdown("<div class='section-header'>Where Customers Go at Renewal</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_renewal_sankey(inv_f), use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 6 — Action Plan
# -----------------------------------------------------------------------------
with tab_action:
    st.markdown("### Strategic Summary")
    st.markdown("Findings and implications, by section.")

    summary = pd.DataFrame([
        ['State of Business',
         'MRR peaked at €211K in January 2026 and has fallen €45K per month since. Churn now exceeds new MRR for the first time.',
         'The business has reached a structural inflection. Without intervention, the trajectory continues.'],
        ['State of Business',
         'First-year renewal rate has been flat at ~55% across all cohorts since 2021, but revenue retention has declined cohort over cohort. The issue is expansion at renewal, not churn rate.',
         'Acquisition volume and churn rate are not the levers. Renewal-time upsell intensity is.'],
        ['Platform Depth',
         'Single-product customers renew at 10.7% and retain 6.8% of Month-0 MRR at expiry. Multi-product (3+) customers renew at 43.6% with 70.8% NRR at expiry.',
         'Moving a customer from one product to two within their first contract is the highest-leverage retention action.'],
        ['Platform Depth',
         'Multi-product customers generate €274 LTV vs €76 for single-product, a 3.6x value gap with retention behaviour to match.',
         'The blueprint exists internally. The challenge is to manufacture more multi-product customers.'],
        ['Country',
         'All 12 markets follow nearly identical retention curves: renewal 23.6% to 25.9%, NRR at expiry 31.6% to 34.3%.',
         'Geography is not the lever. One retention programme applies across all markets.'],
        ['Product',
         'Shared hosting drives 92% of MRR but its NRR at Expiry of 25.7% sits well below the 40.2% headline. Volume engine, not retention engine.',
         'Shared hosting is the front door. It cannot be the retention story alone, it must lead to upsell.'],
        ['Product',
         'WHOIS Privacy and Local ccTLD have the strongest individual product retention. Multi-product customers are over-indexed on these add-ons.',
         'Make WHOIS Privacy and Local ccTLD the default upsell in every shared-hosting onboarding flow.'],
        ['Contract Length',
         '24-month customers have €193 LTV vs €135 for 12-month customers, a 43% premium and €59 more per customer.',
         'Every customer shifted from 12 to 24 months at acquisition is worth €59 more in lifetime value.'],
        ['Contract Length',
         '24-month contract share has fallen from 19.6% (2021) to 14.2% (2026). The contract mix is drifting toward shorter-LTV options.',
         'A 5 percentage point mix-shift recovery to 24-month generates approximately €30K incremental LTV per cohort. The lever is acquisition-time pricing and messaging.'],
        ['Contract Length',
         '50% of 12-month customers do not renew. Most of the rest renew the same product with almost no upgrade flow visible.',
         'The renewal moment is the largest revenue opportunity in the business. It must become an active sales motion, not a passive billing event.'],
    ], columns=['Section', 'Finding', 'Implication'])

    st.dataframe(summary, use_container_width=True, hide_index=True)
