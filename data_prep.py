"""
Data loading, filtering, and KPI computation.
KPI calculations match the team.blue strategic report exactly.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

CUSTOMERS_PARQUET = DATA_DIR / "customers.parquet"
INVOICES_PARQUET = DATA_DIR / "invoices.parquet"
MRR_PARQUET = DATA_DIR / "mrr.parquet"


# =============================================================================
# Data loading (cached)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data():
    if all(p.exists() for p in [CUSTOMERS_PARQUET, INVOICES_PARQUET, MRR_PARQUET]):
        return _load_parquet()
    return _load_synthetic()


def _load_parquet():
    customers = pd.read_parquet(CUSTOMERS_PARQUET)
    invoices = pd.read_parquet(INVOICES_PARQUET)
    mrr = pd.read_parquet(MRR_PARQUET)

    if 'cohort_month' in customers.columns:
        customers['cohort_month'] = pd.to_datetime(customers['cohort_month'])
    if 'invoice_date' in invoices.columns:
        invoices['invoice_date'] = pd.to_datetime(invoices['invoice_date'])
    if 'INVOICE_MONTH' in invoices.columns:
        invoices['INVOICE_MONTH'] = pd.to_datetime(invoices['INVOICE_MONTH'])
    if 'cohort_month' in invoices.columns:
        invoices['cohort_month'] = pd.to_datetime(invoices['cohort_month'])
    if 'mrr_month' in mrr.columns:
        mrr['mrr_month'] = pd.to_datetime(mrr['mrr_month'])

    # Derive subscription_type on mrr if not already there (for contract-length filter)
    if 'subscription_type' not in mrr.columns and 'contract_length_months' in mrr.columns:
        bucket_map = {1: '1 months', 12: '12 months', 24: '24 months',
                      60: '60 months', 120: '120 months'}
        mrr['subscription_type'] = (mrr['contract_length_months']
                                    .map(bucket_map).fillna('Other')
                                    .astype('category'))

    return customers, invoices, mrr


def _load_synthetic():
    from synthetic_data import generate_synthetic_data
    return generate_synthetic_data(n_customers=8000)


# =============================================================================
# Customer anchor lookup (cached, derived from raw data)
# Computed once at app startup. Used by all retention functions to avoid
# repeating the expensive groupby on filtered MRR data.
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_customer_anchors():
    """Returns a small lookup table (one row per customer) with:
      customer_id, cohort_month, cohort_year, first_mrr_month
    Cached as a derivative of load_all_data, so it computes once."""
    customers, _, mrr = load_all_data()

    # cohort_month from customers table (matches sidebar filter source)
    cohort_lookup = (customers[['customer_id', 'cohort_month']]
                     .drop_duplicates('customer_id')
                     .copy())
    cohort_lookup['cohort_month'] = pd.to_datetime(cohort_lookup['cohort_month'])

    # first_mrr_month from full MRR (the customer's true cohort start)
    mrr_dates = mrr[['customer_id', 'mrr_month']].copy()
    mrr_dates['mrr_month'] = pd.to_datetime(mrr_dates['mrr_month'])
    first_mrr = (mrr_dates.groupby('customer_id', observed=True)['mrr_month']
                 .min().rename('first_mrr_month').reset_index())

    anchors = cohort_lookup.merge(first_mrr, on='customer_id', how='outer')
    anchors['cohort_year'] = anchors['cohort_month'].dt.year
    return anchors


# =============================================================================
# Filtering
# =============================================================================

def apply_filters(customers, invoices, mrr, *,
                  date_range=None, countries=None, cohorts=None,
                  product_groups=None, product_subgroups=None,
                  contract_lengths=None, platform_depths=None):

    cust = customers.copy()
    inv = invoices.copy()
    mrr_f = mrr.copy()

    if countries:
        cust = cust[cust['country'].isin(countries)]
        inv = inv[inv['country'].isin(countries)]
        mrr_f = mrr_f[mrr_f['country'].isin(countries)]

    if cohorts:
        cust = cust[cust['cohort_year'].isin(cohorts)]
        inv = inv[inv['cohort_year'].isin(cohorts)]
        mrr_f = mrr_f[mrr_f['cohort_year'].isin(cohorts)]

    if platform_depths:
        cust = cust[cust['platform_depth'].isin(platform_depths)]
        inv = inv[inv['platform_depth'].isin(platform_depths)]
        mrr_f = mrr_f[mrr_f['platform_depth'].isin(platform_depths)]

    if product_groups:
        inv = inv[inv['product_group'].isin(product_groups)]
        mrr_f = mrr_f[mrr_f['product_group'].isin(product_groups)]

    if product_subgroups:
        inv = inv[inv['product_subgroup'].isin(product_subgroups)]
        mrr_f = mrr_f[mrr_f['product_subgroup'].isin(product_subgroups)]

    if contract_lengths:
        cust = cust[cust['subscription_type'].isin(contract_lengths)]
        inv = inv[inv['subscription_type'].isin(contract_lengths)]
        mrr_f = mrr_f[mrr_f['subscription_type'].isin(contract_lengths)]

    if date_range:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mrr_f = mrr_f[(mrr_f['mrr_month'] >= start) & (mrr_f['mrr_month'] <= end)]

    return cust, inv, mrr_f


# =============================================================================
# KPIs — match the strategic report exactly
# =============================================================================

def _avg_mrr_yearly(mrr):
    """Mean of monthly MRR by year, then mean across years 2021-2025."""
    if len(mrr) == 0:
        return 0
    df = mrr.copy()
    df['year'] = pd.to_datetime(df['mrr_month']).dt.year
    monthly = df.groupby(['year', 'mrr_month'])['mrr'].sum().reset_index()
    yearly = monthly.groupby('year')['mrr'].mean()
    yearly = yearly[(yearly.index >= 2021) & (yearly.index <= 2025)]
    return float(yearly.mean()) if len(yearly) else 0


def _arpu_yearly(mrr):
    """Mean of monthly ARPU by year, then mean across years 2021-2025."""
    if len(mrr) == 0:
        return 0
    df = mrr.copy()
    df['year'] = pd.to_datetime(df['mrr_month']).dt.year
    monthly = df.groupby(['year', 'mrr_month']).agg(
        total_mrr=('mrr', 'sum'),
        active=('customer_id', 'nunique'),
    ).reset_index()
    monthly['arpu'] = monthly['total_mrr'] / monthly['active']
    yearly = monthly.groupby('year')['arpu'].mean()
    yearly = yearly[(yearly.index >= 2021) & (yearly.index <= 2025)]
    return float(yearly.mean()) if len(yearly) else 0


def _nrr_at_expiry(invoices):
    """
    Mean across cohort years 2021-2023 of:
        sum(rev_expiry) / sum(rev_0) * 100
    where rev_expiry = invoices in period_num between first_span±2, period_num>0
    """
    if 'first_span' not in invoices.columns or len(invoices) == 0:
        return 0

    inv = invoices.copy()
    inv['cohort_month'] = pd.to_datetime(inv['cohort_month'])
    if 'INVOICE_MONTH' in inv.columns:
        inv['INVOICE_MONTH'] = pd.to_datetime(inv['INVOICE_MONTH'])
    else:
        inv['INVOICE_MONTH'] = pd.to_datetime(inv['invoice_date']).dt.to_period('M').dt.to_timestamp()

    inv['period_num'] = (
        (inv['INVOICE_MONTH'].dt.year - inv['cohort_month'].dt.year) * 12 +
        (inv['INVOICE_MONTH'].dt.month - inv['cohort_month'].dt.month)
    )

    inv = inv[inv['cohort_year'].between(2021, 2023)].copy()
    if len(inv) == 0:
        return 0

    period_0 = (
        inv[inv['period_num'] == 0]
        .groupby(['cohort_year', 'customer_id'], observed=True)
        .agg(rev_0=('amount', 'sum'), span=('first_span', 'max'))
        .reset_index()
    )

    inv_with_span = inv.merge(
        period_0[['customer_id', 'cohort_year', 'span']],
        on=['customer_id', 'cohort_year'], how='inner'
    )
    expiry = inv_with_span[
        (inv_with_span['period_num'] > 0) &
        (inv_with_span['period_num'] >= inv_with_span['span'] - 2) &
        (inv_with_span['period_num'] <= inv_with_span['span'] + 2)
    ]
    period_expiry = (
        expiry.groupby(['cohort_year', 'customer_id'], observed=True)['amount']
        .sum().reset_index().rename(columns={'amount': 'rev_expiry'})
    )

    joined = period_0.merge(
        period_expiry, on=['cohort_year', 'customer_id'], how='left'
    )
    joined['rev_expiry'] = joined['rev_expiry'].fillna(0)

    cohort_totals = joined.groupby('cohort_year', observed=True).agg(
        total_rev_0=('rev_0', 'sum'),
        total_rev_expiry=('rev_expiry', 'sum'),
    )
    cohort_totals['nrr'] = cohort_totals['total_rev_expiry'] / cohort_totals['total_rev_0'] * 100
    return float(cohort_totals['nrr'].mean())


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def compute_kpis(customers, invoices, mrr):
    """Six headline KPIs — matching the report's Cell 19 calculations."""
    if len(mrr) == 0 or len(customers) == 0:
        return {k: 0 for k in ['customers', 'avg_mrr', 'arpu', 'renewal_rate', 'ltv', 'nrr_expiry']}

    # Count customers from invoices so product/subgroup filters narrow the count.
    # (The customers table doesn't have product columns, so customer-table filtering
    # is invariant to product slicers. Invoice-derived count handles every dimension.)
    invoice_customer_ids = invoices['customer_id'].unique()
    n_customers = len(invoice_customer_ids)

    avg_mrr = _avg_mrr_yearly(mrr)
    arpu = _arpu_yearly(mrr)

    # Renewal rate uses customer_ever_renewed flag, scoped to invoice-active customers
    if 'customer_ever_renewed' in customers.columns:
        scoped = customers[customers['customer_id'].isin(invoice_customer_ids)]
        renewal_rate = float(scoped['customer_ever_renewed'].mean() * 100) if len(scoped) else 0
    else:
        # Fallback for synthetic data
        inv_counts = invoices.groupby('customer_id').size()
        renewal_rate = float((inv_counts > 1).mean() * 100)

    ltv = float(invoices['amount'].sum() / max(n_customers, 1))
    nrr_expiry = _nrr_at_expiry(invoices)

    return {
        'customers': n_customers,
        'avg_mrr': avg_mrr,
        'arpu': arpu,
        'renewal_rate': renewal_rate,
        'ltv': ltv,
        'nrr_expiry': nrr_expiry,
    }


def _nrr_at_expiry_by_segment(invoices, dimension):
    """NRR at Expiry per segment — replicates notebook's `kpi_summary` SQL exactly.
    Note: the notebook's join-on-customer (not segment) means customers with
    multiple segments at month 0 contribute their window revenue to each of
    their starting segments. We match that behaviour here for parity."""
    if 'first_span' not in invoices.columns or len(invoices) == 0:
        return {}
    if dimension not in invoices.columns:
        return {}

    inv = invoices.copy()
    inv['cohort_month'] = pd.to_datetime(inv['cohort_month'])
    if 'INVOICE_MONTH' in inv.columns:
        inv['INVOICE_MONTH'] = pd.to_datetime(inv['INVOICE_MONTH'])
    else:
        inv['INVOICE_MONTH'] = pd.to_datetime(inv['invoice_date']).dt.to_period('M').dt.to_timestamp()

    inv['period_num'] = (
        (inv['INVOICE_MONTH'].dt.year - inv['cohort_month'].dt.year) * 12 +
        (inv['INVOICE_MONTH'].dt.month - inv['cohort_month'].dt.month)
    )

    if len(inv) == 0:
        return {}

    period_0 = (
        inv[inv['period_num'] == 0]
        .groupby([dimension, 'customer_id'], observed=True)
        .agg(rev_0=('amount', 'sum'), span=('first_span', 'max'))
        .reset_index()
    )

    # Join on customer_id ONLY (matches notebook SQL). Customers with multiple
    # segments at p0 will have their window invoices duplicated — and the
    # downstream group-by sums those duplicates, mirroring the notebook.
    inv_with_span = inv.merge(
        period_0[['customer_id', 'span']],
        on='customer_id', how='inner'
    )
    expiry = inv_with_span[
        (inv_with_span['period_num'] > 0) &
        (inv_with_span['period_num'] >= inv_with_span['span'] - 2) &
        (inv_with_span['period_num'] <= inv_with_span['span'] + 2)
    ]
    period_expiry = (
        expiry.groupby([dimension, 'customer_id'], observed=True)['amount']
        .sum().reset_index().rename(columns={'amount': 'rev_expiry'})
    )
    joined = period_0.merge(
        period_expiry, on=[dimension, 'customer_id'], how='left'
    )
    joined['rev_expiry'] = joined['rev_expiry'].fillna(0)
    seg_totals = joined.groupby(dimension, observed=True).agg(
        total_rev_0=('rev_0', 'sum'),
        total_rev_expiry=('rev_expiry', 'sum'),
    )
    seg_totals['nrr'] = seg_totals['total_rev_expiry'] / seg_totals['total_rev_0'] * 100
    return seg_totals['nrr'].to_dict()


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def compute_kpi_table(customers, invoices, mrr, dimension):
    """KPI breakdown by a dimension. Matches notebook's `kpi_summary` logic exactly.
    For invoice-level dimensions (product_group, etc.), invoices are filtered by
    the dimension value — not by customer set — so customers with multi-segment
    activity are correctly attributed only to invoices in that segment."""
    if len(mrr) == 0 or dimension not in invoices.columns:
        return pd.DataFrame()

    has_renewal_flag = 'customer_ever_renewed' in customers.columns
    nrr_by_segment = _nrr_at_expiry_by_segment(invoices, dimension)

    rows = []
    for seg in invoices[dimension].dropna().unique():
        seg_inv = invoices[invoices[dimension] == seg]
        seg_customer_ids = seg_inv['customer_id'].unique()
        n = len(seg_customer_ids)
        if n == 0:
            continue

        if dimension in mrr.columns:
            seg_mrr = mrr[mrr[dimension] == seg]
        else:
            seg_mrr = mrr[mrr['customer_id'].isin(seg_customer_ids)]

        # Flat avg of monthly MRR — matches notebook's segment KPI calc.
        # (Headline KPIs in compute_kpis use year-weighted; segment tables don't.)
        seg_monthly = seg_mrr.groupby('mrr_month')['mrr'].sum()
        seg_active = seg_mrr.groupby('mrr_month')['customer_id'].nunique()
        seg_avg_mrr = float(seg_monthly.mean()) if len(seg_monthly) else 0
        seg_arpu = float((seg_monthly / seg_active).mean()) if len(seg_monthly) else 0

        seg_customers = customers[customers['customer_id'].isin(seg_customer_ids)]
        if has_renewal_flag:
            renewal = float(seg_customers['customer_ever_renewed'].mean() * 100)
        else:
            inv_counts = seg_inv.groupby('customer_id').size()
            renewal = float((inv_counts > 1).mean() * 100)

        # LTV = revenue from THIS segment / customers in this segment
        ltv = float(seg_inv['amount'].sum() / max(n, 1))
        nrr = nrr_by_segment.get(seg)

        rows.append({
            'Segment': str(seg),
            'Customers': n,
            'Avg MRR/month (€)': round(seg_avg_mrr, 0),
            'ARPU (€/month)': round(seg_arpu, 2),
            'Renewal Rate %': round(renewal, 1),
            'Avg LTV (€)': round(ltv, 2),
            'NRR at Expiry %': round(nrr, 1) if nrr is not None and not pd.isna(nrr) else None,
        })

    return pd.DataFrame(rows).sort_values('Avg MRR/month (€)', ascending=False)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def compute_retention_curve(mrr, customers=None, group_col=None, group_value=None):
    """% of Month-0 MRR retained at each month-since-acquisition, by cohort.
    Matches the notebook's plot_revenue_retention_curve calculation exactly:
    period_num = (mrr_month - cohort_month) using to_period('M') arithmetic.
    """
    df = mrr.copy()
    if group_col and group_value:
        df = df[df[group_col] == group_value]
    if len(df) == 0:
        return pd.DataFrame()

    df['mrr_month'] = pd.to_datetime(df['mrr_month'])

    # Get cohort_month for each customer — from MRR if present, else merge from customers
    if 'cohort_month' not in df.columns:
        if customers is not None and 'cohort_month' in customers.columns:
            cohort_lookup = (customers[['customer_id', 'cohort_month']]
                             .drop_duplicates('customer_id'))
            df = df.merge(cohort_lookup, on='customer_id', how='left')
        else:
            # Fallback: synthetic data, use first MRR month per customer
            first_month = df.groupby('customer_id')['mrr_month'].min().rename('cohort_month')
            df = df.merge(first_month, on='customer_id', how='left')

    df = df.dropna(subset=['cohort_month'])
    df['cohort_month'] = pd.to_datetime(df['cohort_month'])

    # Vectorized month difference (much faster than .apply on Period arithmetic)
    df['months_since'] = (
        (df['mrr_month'].dt.year - df['cohort_month'].dt.year) * 12 +
        (df['mrr_month'].dt.month - df['cohort_month'].dt.month)
    )

    df = df[df['months_since'] >= 0]  # drop any pre-cohort rows
    df = df[df['months_since'] <= 36]  # cap at 36 months like the notebook

    agg = df.groupby(['cohort_year', 'months_since'], observed=True)['mrr'].sum().reset_index()
    m0 = agg[agg['months_since'] == 0].set_index('cohort_year')['mrr']
    agg['pct_retained'] = agg.apply(
        lambda r: (r['mrr'] / m0[r['cohort_year']] * 100) if r['cohort_year'] in m0.index else 0,
        axis=1
    )
    return agg


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def compute_nrr_cohort_curve_clean(mrr, customers=None, max_months=36,
                                    maturity_threshold=0.80):
    """Stacking-corrected NRR (Net Revenue Retention) by acquisition cohort year.

    Computes NRR per *monthly* sub-cohort (so each curve hits its M12 cleanly),
    then averages within each year weighted by sub-cohort M0 MRR.
    Removes the yearly-cohort blending artifact where customers joining
    different months of the same year contribute to the same period_num
    despite being at different lifecycle stages.

    Right-censoring: months where <maturity_threshold of the year's
    sub-cohorts have enough observation time are excluded.
    """
    df = mrr.copy()
    if len(df) == 0:
        return pd.DataFrame()

    df['mrr_month'] = pd.to_datetime(df['mrr_month'])

    # Use cached customer anchor lookup (cohort_month, cohort_year, first_mrr_month)
    anchors = get_customer_anchors()
    df = df.drop(columns=[c for c in ('cohort_month', 'cohort_year', 'first_mrr_month')
                          if c in df.columns])
    df = df.merge(anchors, on='customer_id', how='left')
    df = df.dropna(subset=['cohort_month', 'first_mrr_month'])

    # Vectorized month difference
    df['period_num'] = (
        (df['mrr_month'].dt.year - df['first_mrr_month'].dt.year) * 12 +
        (df['mrr_month'].dt.month - df['first_mrr_month'].dt.month)
    )

    # Vectorized YYYY-MM string (faster than to_period().astype(str))
    df['cohort_ym'] = (
        df['cohort_month'].dt.year.astype(str) + '-' +
        df['cohort_month'].dt.month.astype(str).str.zfill(2)
    )

    df = df[df['cohort_year'].between(2021, 2025)]
    df = df[(df['period_num'] >= 0) & (df['period_num'] <= max_months)]

    # MRR per (monthly cohort, period_num)
    sub = (df.groupby(['cohort_ym', 'cohort_year', 'period_num'], observed=True)['mrr']
             .sum().reset_index())

    # M0 MRR per monthly cohort (denominator)
    m0 = (sub[sub['period_num'] == 0][['cohort_ym', 'mrr']]
              .rename(columns={'mrr': 'mrr_m0'}))
    sub = sub.merge(m0, on='cohort_ym')
    sub = sub[sub['mrr_m0'] > 0]
    if len(sub) == 0:
        return pd.DataFrame()

    sub['nrr'] = sub['mrr'] / sub['mrr_m0'] * 100

    # Aggregate to yearly cohort weighted by sub-cohort M0 MRR
    def _weighted_mean(g):
        return (g['nrr'] * g['mrr_m0']).sum() / g['mrr_m0'].sum()

    yearly = (sub.groupby(['cohort_year', 'period_num'], observed=True)
                 .apply(_weighted_mean)
                 .reset_index(name='pct_retained'))

    # Right-censoring per yearly cohort
    data_max = df['mrr_month'].max()
    sub_starts = sub[['cohort_ym', 'cohort_year']].drop_duplicates()
    sub_starts['start_period'] = pd.PeriodIndex(sub_starts['cohort_ym'], freq='M')
    data_max_period = data_max.to_period('M')
    sub_starts['months_avail'] = (data_max_period - sub_starts['start_period']).apply(lambda x: x.n)

    def _maturity(year, period):
        starts = sub_starts[sub_starts['cohort_year'] == year]
        if len(starts) == 0:
            return 0
        return (starts['months_avail'] >= period).mean()

    yearly['mature_pct'] = yearly.apply(
        lambda r: _maturity(r['cohort_year'], r['period_num']), axis=1)
    yearly = yearly[yearly['mature_pct'] >= maturity_threshold]

    return yearly[['cohort_year', 'period_num', 'pct_retained']].rename(
        columns={'period_num': 'months_since'})


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def compute_logo_retention_curve(mrr, customers=None, group_col=None, group_value=None):
    """% of cohort customers still active at each month-since-acquisition.
    Bounded at 100%. Each customer counts as 1.

    Time-axis uses each customer's first MRR month as their personal M0
    (not invoice cohort_month, which can be a month earlier than MRR start).
    This guarantees every customer has an M0 row, so the curve starts at 100%.
    Cohort assignment (which year they belong to) still uses the original
    cohort_year column from invoices.

    Adds 'mature_pct' column = % of cohort customers who had enough observation
    time to *potentially* reach this months_since milestone. Cells where
    mature_pct is low are right-censored — recent cohorts haven't had time.
    """
    df = mrr.copy()
    if group_col and group_value:
        df = df[df[group_col] == group_value]
    if len(df) == 0:
        return pd.DataFrame()

    df['mrr_month'] = pd.to_datetime(df['mrr_month'])
    data_max_month = df['mrr_month'].max()

    # Use cached customer anchor lookup (avoids per-call groupby on filtered MRR)
    anchors = get_customer_anchors()
    df = df.drop(columns=[c for c in ('cohort_month', 'cohort_year', 'first_mrr_month')
                          if c in df.columns])
    df = df.merge(anchors, on='customer_id', how='left')
    df = df.dropna(subset=['first_mrr_month'])

    df['months_since'] = (
        (df['mrr_month'].dt.year - df['first_mrr_month'].dt.year) * 12 +
        (df['mrr_month'].dt.month - df['first_mrr_month'].dt.month)
    )

    df = df[df['months_since'] >= 0]
    df = df[df['months_since'] <= 36]

    agg = (df.groupby(['cohort_year', 'months_since'], observed=True)['customer_id']
             .nunique().reset_index().rename(columns={'customer_id': 'active_customers'}))

    m0 = agg[agg['months_since'] == 0].set_index('cohort_year')['active_customers']
    agg['pct_retained'] = agg.apply(
        lambda r: (r['active_customers'] / m0[r['cohort_year']] * 100)
                  if r['cohort_year'] in m0.index else 0,
        axis=1
    )

    # Compute % of cohort that had observation time to reach each milestone.
    # For a customer who started in month X, they could reach milestone N
    # if X + N months <= data_max_month.
    cust_starts = df.drop_duplicates('customer_id')[['customer_id', 'cohort_year', 'first_mrr_month']]
    cust_starts['months_available'] = (
        cust_starts['first_mrr_month'].apply(lambda d:
            ((data_max_month.year - d.year) * 12 + (data_max_month.month - d.month)))
    )

    mature_records = []
    for cy, group in cust_starts.groupby('cohort_year', observed=True):
        cohort_size = len(group)
        for ms in agg[agg['cohort_year'] == cy]['months_since'].unique():
            mature_count = (group['months_available'] >= ms).sum()
            mature_records.append({
                'cohort_year': cy, 'months_since': ms,
                'mature_pct': mature_count / cohort_size * 100 if cohort_size else 0,
            })
    mature_df = pd.DataFrame(mature_records)
    agg = agg.merge(mature_df, on=['cohort_year', 'months_since'], how='left')
    return agg


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def compute_logo_retention_monthly(mrr, customers=None, max_months=36, start_year=2021):
    """% of cohort customers still active at each month-since-acquisition,
    bucketed by *monthly* acquisition cohort (instead of yearly).

    Returns a DataFrame with:
      cohort_ym     : 'YYYY-MM' string (e.g. '2021-03')
      n_t0          : customers in the cohort at M0
      months_avail  : months of observation available for this cohort
      m0..mN        : retention % at each month (NaN if right-censored)
    """
    df = mrr.copy()
    if len(df) == 0:
        return pd.DataFrame()

    df['mrr_month'] = pd.to_datetime(df['mrr_month'])

    # Use cached customer anchor lookup (cohort_month, cohort_year, first_mrr_month)
    # This avoids the expensive groupby + merge that would otherwise run every call
    anchors = get_customer_anchors()
    df = df.drop(columns=[c for c in ('cohort_month', 'cohort_year', 'first_mrr_month')
                          if c in df.columns])
    df = df.merge(anchors, on='customer_id', how='left')
    df = df.dropna(subset=['cohort_month', 'first_mrr_month'])

    # Vectorized month difference
    df['period_num'] = (
        (df['mrr_month'].dt.year - df['first_mrr_month'].dt.year) * 12 +
        (df['mrr_month'].dt.month - df['first_mrr_month'].dt.month)
    )

    # Vectorized YYYY-MM string
    df['cohort_ym'] = (
        df['cohort_month'].dt.year.astype(str) + '-' +
        df['cohort_month'].dt.month.astype(str).str.zfill(2)
    )

    df = df[df['cohort_year'] >= start_year]
    df = df[(df['period_num'] >= 0) & (df['period_num'] <= max_months)]

    if len(df) == 0:
        return pd.DataFrame()

    data_max = df['mrr_month'].max()
    data_max_period = data_max.to_period('M')

    # M0 customer count per monthly cohort
    m0 = (df[df['period_num'] == 0]
              .groupby('cohort_ym')['customer_id']
              .nunique()
              .rename('n_t0'))

    counts = (df.groupby(['cohort_ym', 'period_num'], observed=True)['customer_id']
                 .nunique()
                 .reset_index(name='n_active'))
    counts = counts.merge(m0, on='cohort_ym')
    counts['pct'] = counts['n_active'] / counts['n_t0'] * 100

    pivot = counts.pivot(index='cohort_ym', columns='period_num', values='pct')
    pivot = pivot.sort_index()

    # Months of observation available per cohort
    cohort_periods = pd.PeriodIndex(pivot.index, freq='M')
    months_avail = pd.Series(
        (data_max_period - cohort_periods).map(lambda x: x.n),
        index=pivot.index, name='months_avail'
    )

    # Right-censor: blank cells where cohort hasn't reached that period yet
    for p in pivot.columns:
        mask = months_avail < p
        pivot.loc[mask, p] = pd.NA

    out = pivot.copy()
    out.insert(0, 'n_t0', m0)
    out.insert(1, 'months_avail', months_avail)
    out = out.reset_index()
    return out
