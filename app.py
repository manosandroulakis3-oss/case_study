"""
Case Study Cohort Analysis — Streamlit Dashboard
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
    page_title="Case Study | Cohort Analysis",
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
        "Cells marked with a dash are right-censored (cohort hasn't had enough observation time).</div>",
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


def render_text_table(df):
    """Render a DataFrame as an HTML table with proper text wrapping.
    Use for tables with long-text cells (st.dataframe truncates instead of wraps)."""
    html = """
    <style>
    table.text-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    table.text-table th {
        background: #f3f4f6;
        text-align: left;
        padding: 10px 12px;
        font-weight: 600;
        color: #374151;
        border-bottom: 2px solid #e5e7eb;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    table.text-table td {
        padding: 10px 12px;
        vertical-align: top;
        border-bottom: 1px solid #f3f4f6;
        line-height: 1.5;
        color: #1f2937;
    }
    table.text-table tbody tr:hover { background: #fafafa; }
    </style>
    <table class="text-table"><thead><tr>
    """
    for c in df.columns:
        html += f"<th>{c}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for c in df.columns:
            val = str(row[c]).replace('\n', '<br>')
            html += f"<td>{val}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


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


# Clear-all-filters button
def _clear_filters():
    for k in ('flt_country', 'flt_cohort', 'flt_pgroup', 'flt_psub',
             'flt_clen', 'flt_pdepth'):
        if k in st.session_state:
            del st.session_state[k]


st.sidebar.button(
    "Clear all filters", on_click=_clear_filters,
    use_container_width=True, type="secondary",
)

date_range = None  # Date range filter removed — full data range is always used

_country_options = [c for c in sorted(customers['country'].unique()) if c != 'Estonia']
countries = st.sidebar.multiselect(
    "Country", _country_options,
    default=None, placeholder="All countries", key='flt_country',
)

cohorts = st.sidebar.multiselect(
    "Acquisition cohort", sorted(customers['cohort_year'].unique()),
    default=None, placeholder="All cohorts", key='flt_cohort',
)

product_groups = st.sidebar.multiselect(
    "Product group", sorted(invoices['product_group'].unique()),
    default=None, placeholder="All groups", key='flt_pgroup',
)

product_subgroups = st.sidebar.multiselect(
    "Product subgroup", sorted(invoices['product_subgroup'].unique()),
    default=None, placeholder="All subgroups", key='flt_psub',
)

_sub_order = ['1 months', '12 months', '24 months', '60 months', '120 months', 'Other']
_sub_available = [s for s in _sub_order if s in customers['subscription_type'].unique()]
contract_lengths = st.sidebar.multiselect(
    "Contract length",
    _sub_available,
    default=None, placeholder="All lengths", key='flt_clen',
)

platform_depths = st.sidebar.multiselect(
    "Platform depth", sorted(customers['platform_depth'].unique()),
    default=None, placeholder="All depths", key='flt_pdepth',
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
st.markdown("<h1 class='dashboard-title'>Case Study · Cohort Analysis</h1>",
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
tab_state, tab_depth, tab_country, tab_product, tab_contract, tab_action, tab_notes = st.tabs([
    "📈 State of Business",
    "🔢 Platform Depth",
    "🌍 Country",
    "📦 Product",
    "📅 Contract Length",
    "🎯 Action Plan",
    "📖 Notes",
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
        retention_title = f'Cohort Retention: {selected_countries[0]}'
    elif len(selected_countries) <= 3:
        retention_title = f'Cohort Retention: {", ".join(map(str, selected_countries))}'
    else:
        retention_title = f'Cohort Retention: {len(selected_countries)} countries selected'

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

    render_text_table(summary)


# -----------------------------------------------------------------------------
# Tab 7 — Notes
# -----------------------------------------------------------------------------
with tab_notes:
    st.markdown("### KPI Definitions")
    st.markdown(
        "<div class='caption'>Each KPI shown in the headline strip and segment tables, "
        "with formula and interpretation.</div>",
        unsafe_allow_html=True
    )

    kpi_defs = pd.DataFrame([
        ['Customers',
         'Count of unique customer IDs in the filtered dataset',
         'How many customers are reflected in the current view. Filtering by product narrows this to customers who bought in that segment.'],
        ['Avg MRR / Month',
         'Mean of monthly MRR totals, year-weighted across 2021-2025',
         'Recurring revenue earning power. Computed as the average of yearly means rather than a flat average to avoid bias from cohort growth.'],
        ['ARPU',
         'Average revenue per user per month, year-weighted across 2021-2025',
         'How much each active customer contributes per month on average. Same year-weighting as Avg MRR.'],
        ['Renewal Rate',
         '% of customers who ever repurchased the same product after first month',
         'Logo-level retention signal: did this customer come back, regardless of spend? Uses the customer_ever_renewed flag built in the data prep.'],
        ['Avg LTV',
         'Total revenue ÷ unique customers',
         'Lifetime value to date. Includes all revenue across full observation window.'],
        ['NRR at Expiry',
         'Sum of revenue in renewal window (first_span ± 2 months) ÷ Sum of revenue at Month 0',
         'Net revenue retention measured at the renewal moment for cohorts 2021-2023 (mature cohorts only). Above 100% = expansion outweighs churn at renewal.'],
    ], columns=['KPI', 'Formula', 'Interpretation'])
    render_text_table(kpi_defs)

    st.markdown("### Why Revenue Retention can go above 100%")
    st.markdown(
        """
The Revenue Retention curve in the State of Business tab shows values above 100% for some cohorts, especially around Months 12 and 24. This is **expected and meaningful**, not a bug.

**Net Revenue Retention (NRR)** measures *how much money a cohort generates over time*, not *how many customers stayed*. It includes:

- **Retained MRR**, customers who keep paying
- **Expansion MRR**, customers who upgrade or add products (positive contribution)
- **Contraction MRR**, customers who downgrade (negative)
- **Churned MRR**, customers who leave (negative)

When expansion exceeds churn + contraction, NRR exceeds 100%. This data shows it clearly at Month 12. Many customers renew their annual contract *with an upsell* (more domains, hosting upgrades, WHOIS Privacy add-on), so the cohort's total spend in Month 12 is higher than at acquisition.

This is the standard SaaS metric. Public companies like Snowflake (~140%) and Datadog (~130%) report NRR above 100%. It's how investors evaluate whether a business is growing existing accounts.

**A note on cohort stacking.** Yearly cohorts blend customers who joined throughout the year. A March 2021 joiner reaches their personal Month 12 in March 2022; a December 2021 joiner reaches it in December 2022. The Month 12 spike reflects expansion plus renewal-with-upsell aggregated across all 12 monthly sub-cohorts. The cross-cohort comparison (2021 vs 2022 vs 2023) remains valid because the same blending applies to every cohort year.

For a churn-only view bounded at 100%, see the **Customer Retention** table directly above the curve. It tracks the fraction of cohort customers still active, ignoring spend changes.
        """
    )

    st.markdown("### Method Notes")
    st.markdown(
        """
**Customer count by filter.** The Customers KPI is derived from the filtered invoices table. This way, filtering by Product Group narrows the count to customers who bought in that group. Customer-level filters (Country, Cohort, Platform Depth, Contract Length) work directly on the customers table.

**Cohort assignment.** A customer's cohort_year is the year of their first invoice (cohort_month). This is fixed per customer, regardless of subsequent activity.

**Estonia exclusion.** Estonia has only 18 customers in the dataset, too few to draw conclusions from. It is excluded from the Country tab's slicer and KPI table, but counted in the headline KPIs.

**Right-censoring in the Customer Retention table.** Cells are marked with a dash when fewer than 80% of the cohort had observation time to potentially reach that milestone. For example, Cohort 2025's Month 24 column shows a dash because most Cohort 2025 customers haven't existed for 24 months yet; the data only runs through April 2026.

**NRR at Expiry, mature-cohort only.** The headline NRR at Expiry KPI uses cohorts 2021-2023 only, where customers have had at least one chance to reach their renewal window. Younger cohorts are excluded to avoid skew from incomplete observation. Segment-level NRR tables use all cohorts.

**Year-weighted vs flat means.** The headline Avg MRR and ARPU use year-weighted means (mean of yearly means across 2021-2025). Segment-level KPI tables use flat means of monthly totals, matching the strategic report's definition.
        """
    )

    st.markdown("### Data")
    st.markdown(
        """
**Source.** The dashboard reads from three Parquet files: customers (one row per customer with cohort and segment attributes), invoices (one row per billing event), and MRR (one row per customer-product-month).

**Date range.** Data spans March 2019 through April 2026. The full range is used for KPI calculations to match the strategic report's definitions. Time-series charts visually start at January 2021 for clarity.

**Customer count.** 70,938 unique customers across 12 European markets.

**Filtered counts may differ slightly from notebook tables** by single digits (under 0.02%) due to edge cases in segment derivation (e.g., customers who shifted between platform depths over time).
        """
    )
