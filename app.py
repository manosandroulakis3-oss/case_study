"""
Case Study Cohort Analysis — Streamlit Dashboard
Built to mirror the strategic report. BI-tool styling.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from data_prep import (
    load_all_data, apply_filters, compute_kpis,
    compute_kpi_table, compute_retention_curve, compute_logo_retention_curve,
    compute_nrr_cohort_curve_clean, compute_logo_retention_monthly,
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


def render_logo_retention_monthly_table(monthly_df, max_months=36):
    """Render the monthly cohort retention heatmap-style table.
    Expects DataFrame from compute_logo_retention_monthly with columns:
    cohort_ym, n_t0, months_avail, then m0..mN as period_num columns."""
    if monthly_df is None or len(monthly_df) == 0:
        st.info('No data for the current filter selection.')
        return

    def class_for_pct(v):
        if pd.isna(v):
            return 'rb'  # right-censored / blank
        if v >= 90:
            return 'g3'  # dark green
        if v >= 70:
            return 'g2'  # light green
        if v >= 50:
            return 'y'   # amber
        if v >= 30:
            return 'o'   # orange
        return 'r'       # red

    period_cols = [c for c in monthly_df.columns
                   if c not in ('cohort_ym', 'n_t0', 'months_avail')]
    period_cols = sorted([int(c) for c in period_cols])
    period_cols = [p for p in period_cols if p <= max_months]

    # CSS injected once
    css = """
    <style>
    .crh-wrap { overflow-x: auto; max-height: 640px; overflow-y: auto;
                border: 1px solid #e5e7eb; border-radius: 6px; }
    .crh { border-collapse: collapse; font-size: 11px;
           font-family: -apple-system, BlinkMacSystemFont, sans-serif;
           width: max-content; }
    .crh th { padding: 8px 6px; text-align: right; background: #34495e;
              color: white; position: sticky; top: 0; z-index: 3; }
    .crh th.cohort { padding: 8px 10px; text-align: left; position: sticky;
                     left: 0; z-index: 4; }
    .crh td { padding: 5px 6px; border-bottom: 1px solid #eee;
              text-align: right; }
    .crh td.cohort { padding: 6px 10px; background: #f8f9fa; font-weight: 600;
                     position: sticky; left: 0; z-index: 1; }
    .crh td.n { padding: 6px 10px; color: #555; }
    .crh tr.j td { border-top: 2px solid #34495e; }
    .crh td.g3 { background: #d4edda; color: #155724; }
    .crh td.g2 { background: #e2f0d9; color: #2d5e1e; }
    .crh td.y  { background: #fff3cd; color: #856404; }
    .crh td.o  { background: #ffe5b4; color: #6e3a06; }
    .crh td.r  { background: #f8d7da; color: #721c24; }
    .crh td.rb { background: #fafafa; color: #ccc; }
    </style>
    """

    parts = [css, '<div class="crh-wrap"><table class="crh"><thead><tr>',
             '<th class="cohort">Cohort</th><th>N</th>']
    for p in period_cols:
        parts.append(f'<th>M{p}</th>')
    parts.append('</tr></thead><tbody>')

    for _, row in monthly_df.iterrows():
        cohort_ym = str(row['cohort_ym'])
        n_t0 = int(row['n_t0']) if pd.notna(row['n_t0']) else 0
        avail = row['months_avail']
        is_jan = cohort_ym.endswith('-01')
        tr_class = ' class="j"' if is_jan else ''
        parts.append(f'<tr{tr_class}>')
        parts.append(f'<td class="cohort">{cohort_ym}</td>')
        parts.append(f'<td class="n">{n_t0:,}</td>')
        for p in period_cols:
            val = row.get(p, pd.NA)
            if pd.notna(avail) and p > avail:
                parts.append('<td class="rb">.</td>')
            elif pd.isna(val):
                parts.append('<td class="rb">.</td>')
            else:
                cls = class_for_pct(float(val))
                parts.append(f'<td class="{cls}">{float(val):.0f}%</td>')
        parts.append('</tr>')

    parts.append('</tbody></table></div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


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
tab_state, tab_depth, tab_country, tab_product, tab_contract, tab_action, tab_deepdive, tab_notes = st.tabs([
    "📈 State of Business",
    "🔢 Platform Depth",
    "🌍 Country",
    "📦 Product",
    "📅 Contract Length",
    "🎯 Action Plan",
    "🔬 Cohort Deep Dive",
    "📖 Notes",
])


# -----------------------------------------------------------------------------
# Tab 1 — State of the Business
# -----------------------------------------------------------------------------
with tab_state:
    # Headline chart full-width
    st.plotly_chart(chart_mrr_over_time(mrr_chart), use_container_width=True, key='c_mrr_overall')

    # ARPU + NRR side by side
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_arpu_over_time(mrr_chart), use_container_width=True, key='c_arpu')
    with c2:
        st.plotly_chart(chart_nrr_over_time(mrr_chart), use_container_width=True, key='c_nrr')

    # MRR movement gets its own row — bars need horizontal space
    st.plotly_chart(chart_mrr_movement(mrr_chart), use_container_width=True, key='c_mrr_movement')

    # Cohort retention — table first (milestones), then curve (full shape)
    render_logo_retention_table(mrr_chart, customers)

    # NRR cohort curve — stacking-corrected (monthly sub-cohorts averaged within year)
    ret = compute_nrr_cohort_curve_clean(mrr_chart, customers)
    st.plotly_chart(
        chart_retention_curve(ret, title='NRR (Net Revenue Retention) by acquisition cohort year'),
        use_container_width=True, key='c_retention_overall'
    )


# -----------------------------------------------------------------------------
# Tab 2 — Platform Depth
# -----------------------------------------------------------------------------
with tab_depth:
    st.markdown("<div class='section-header'>KPIs by Platform Depth</div>",
                unsafe_allow_html=True)
    table = compute_kpi_table(cust_f, inv_f, mrr_f, 'platform_depth')
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.plotly_chart(chart_mrr_by_dimension(mrr_chart, 'platform_depth',
                    'MRR by Platform Depth'), use_container_width=True, key='c_mrr_depth')

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
                    use_container_width=True, key='c_mrr_country')

    # Retention table — for selected countries (Estonia excluded)
    render_logo_retention_table(mrr_chart_c, customers)

    # NRR cohort curve — stacking-corrected
    selected_countries = sorted([c for c in mrr_chart_c['country'].unique() if c == c])
    if len(selected_countries) == 1:
        retention_title = f'NRR by Cohort: {selected_countries[0]}'
    elif len(selected_countries) <= 3:
        retention_title = f'NRR by Cohort: {", ".join(map(str, selected_countries))}'
    else:
        retention_title = f'NRR by Cohort: {len(selected_countries)} countries selected'

    ret = compute_nrr_cohort_curve_clean(mrr_chart_c, customers)
    st.plotly_chart(chart_retention_curve(ret, retention_title),
                    use_container_width=True, key='c_retention_country')


# -----------------------------------------------------------------------------
# Tab 4 — Product
# -----------------------------------------------------------------------------
with tab_product:
    st.markdown("<div class='section-header'>KPIs by Product Group</div>",
                unsafe_allow_html=True)
    table = compute_kpi_table(cust_f, inv_f, mrr_f, 'product_group')
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.plotly_chart(chart_mrr_by_dimension(mrr_chart, 'product_group', 'MRR by Product Group'),
                    use_container_width=True, key='c_mrr_product')

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
        use_container_width=True, key='c_mrr_contract'
    )

    # Retention table — for whatever's selected in the filter
    render_logo_retention_table(mrr_chart, customers)

    # Sankey on its own row — needs full width
    st.markdown("<div class='section-header'>Where Customers Go at Renewal</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_renewal_sankey(inv_f), use_container_width=True, key='c_sankey')


# -----------------------------------------------------------------------------
# Tab 6 — Action Plan
# -----------------------------------------------------------------------------
with tab_action:
    st.markdown("### Insights and Recommendations")
    st.markdown("Key findings from the analysis with corresponding recommendations, by section.")

    summary = pd.DataFrame([
        ['State of the Business',
         'MRR peaked at €211K in January 2026 and fell €45K in three months. Churned MRR exceeds new MRR in early 2026.',
         'Treat the contraction as the core priority for the next planning cycle. The trajectory has not self-corrected over four years and is now visible across every grouping dimension.'],
        ['State of the Business',
         'First-year retention sits between 54% and 58% across all four mature cohorts. Revenue retention at the same milestones has declined cohort over cohort.',
         'Direct retention efforts at revenue per renewing customer rather than at lifting the headline renewal rate. The renewal rate has been stable since 2021 while revenue retention has fallen cohort over cohort.'],
        ['Platform Depth',
         'Single-product customers (46.5% of the base) renew at 10.7% with €76 in average LTV. Multi-product (3+) customers renew at 43.6% with €274 in average LTV.',
         'Make a second product purchase within the first contract year the primary retention objective. The 3.6x LTV gap between single and multi-product customers is the largest single-step value uplift in the data.'],
        ['Platform Depth',
         'Multi-product cohorts retain above 100% of Month-0 MRR through Month 48. They are the only platform-depth segment whose mature cohorts exceed their starting revenue beyond the first renewal window.',
         'Treat the multi-product cohort as the target retention model for the rest of the base. A working pattern already exists internally.'],
        ['Country',
         'All 12 markets follow nearly identical retention curves. Renewal rates span 21.2% (Poland) to 25.9% (France), a 4.7 percentage point range.',
         'Run a single centralised retention programme across all 12 markets. The retention curves are too close to justify country-level differentiation.'],
        ['Product',
         'Shared Hosting contributes €113,554 in average monthly MRR, the largest share of any product group. Its NRR at expiry of 25.7% sits below the levels seen in WHOIS Privacy, Local ccTLD and Other TLDs.',
         'Treat Shared Hosting as the acquisition front door rather than the retention engine. Build a structured path from Shared Hosting to a domain product within the first contract year.'],
        ['Product',
         'WHOIS Privacy and Local ccTLD show the strongest mature-cohort retention. The 2021 cohort returns above 240% of Month-0 MRR at Month 12 and above 145% at Month 36.',
         'Make WHOIS Privacy and Local ccTLD the default upsell in the Shared Hosting onboarding flow. They are the only products whose mature cohorts grow above their starting revenue.'],
        ['Contract Length',
         '24-month customers post €194 in average LTV against €135 for 12-month customers, a 43% gap. ARPU is 14% higher.',
         'Move 24-month to the default contract option at acquisition. The €59 per-customer LTV gap is a pricing and UX lever, not a product change.'],
        ['Contract Length',
         '24-month contract share at acquisition has fallen from 19.6% in 2021 to 14.2% in 2026.',
         'Reverse the drift toward shorter contracts in the acquisition flow. The current mix is silently reducing per-cohort LTV.'],
        ['Contract Length',
         '27,954 of 55,850 customers (50%) starting on 12-month contracts are lost at the first renewal. The flow from 12-month into longer contracts at renewal is a small fraction of the total.',
         'Treat the 12-month renewal as an active sales touchpoint. Build a structured 24-month upgrade offer and a retention intervention ahead of expiry.'],
    ], columns=['Section', 'Insight', 'Recommendation'])

    render_text_table(summary)


# -----------------------------------------------------------------------------
# Tab 7 — Cohort Deep Dive
# -----------------------------------------------------------------------------
with tab_deepdive:
    st.markdown("### Customer Retention by Monthly Acquisition Cohort")
    st.markdown(
        "<div class='caption'>Each row is a monthly acquisition cohort. "
        "Columns show the percentage of cohort customers still active at each "
        "month-since-acquisition. Right-censored cells (cohort hasn't had time to "
        "reach that milestone) are blank. Maturity threshold: 80%.</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Use this view to spot seasonal patterns, individual underperforming "
        "cohorts, or drift in retention quality over time. Yearly cohorts in "
        "other tabs aggregate across these monthly buckets."
    )

    monthly_df = compute_logo_retention_monthly(mrr_chart, customers, max_months=36)
    render_logo_retention_monthly_table(monthly_df, max_months=36)


# -----------------------------------------------------------------------------
# Tab 8 — Notes
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

    st.markdown("### Why NRR can go above 100%")
    st.markdown(
        """
The NRR curve in the State of Business and Country tabs shows values above 100% at the M12 and M24 renewal moments. This is **expected and meaningful**, not a bug.

**Net Revenue Retention (NRR)** measures *how much money a cohort generates over time*, not *how many customers stayed*. It includes:

- **Retained MRR**, customers who keep paying
- **Expansion MRR**, customers who upgrade or add products (positive contribution)
- **Contraction MRR**, customers who downgrade (negative)
- **Churned MRR**, customers who leave (negative)

When expansion exceeds churn + contraction, NRR exceeds 100%. This data shows it clearly at Month 12. Many customers renew their annual contract *with an upsell* (more domains, hosting upgrades, WHOIS Privacy add-on), so the cohort's total spend in Month 12 is higher than at acquisition.

This is the standard SaaS metric. Public companies like Snowflake (~140%) and Datadog (~130%) report NRR above 100%. It's how investors evaluate whether a business is growing existing accounts.
        """
    )
