"""
Plotly chart factories. Every chart shares a single style template
so the dashboard reads as a coordinated set, not a wall of plots.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# =============================================================================
# Palette
# =============================================================================
TEAMBLUE = '#1E5AB9'
INK = '#1F2937'
MUTED = '#6B7280'
GRID = '#F3F4F6'
GREEN = '#16A34A'
RED = '#DC2626'

# Cohort palette — distinct colors so cohorts read as separate series
COHORT_PALETTE = {
    2021: '#1E5AB9',  # team.blue blue
    2022: '#B23A2D',  # terracotta
    2023: '#0F8A6F',  # teal
    2024: '#7C3AED',  # purple
    2025: '#E67E22',  # orange
}

# Categorical palette — used for countries, products, contract lengths
CAT_PALETTE = ['#1E5AB9', '#13315C', '#B23A2D', '#B8893F', '#2D5F3F',
               '#6B7280', '#5A8DD8', '#8E44AD', '#E67E22', '#16A085',
               '#34495E', '#C0392B']


# =============================================================================
# Layout template
# =============================================================================
def base_layout(title=None, height=380, ytitle=None, xtitle=None, legend=True):
    return dict(
        title=dict(text=title, font=dict(size=14, color=INK, family='Inter, sans-serif'),
                   x=0.01, xanchor='left') if title else None,
        height=height,
        margin=dict(l=55, r=25, t=45 if title else 20, b=80),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=11, color=INK),
        xaxis=dict(
            title=dict(text=xtitle, font=dict(size=11, color=MUTED)) if xtitle else None,
            gridcolor=GRID, showgrid=True, zeroline=False,
            tickfont=dict(size=10, color=MUTED), linecolor='#E5E7EB',
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text=ytitle, font=dict(size=11, color=MUTED)) if ytitle else None,
            gridcolor=GRID, showgrid=True, zeroline=False,
            tickfont=dict(size=10, color=MUTED), linecolor='#E5E7EB',
            automargin=True,
        ),
        showlegend=legend,
        legend=dict(
            orientation='h', yanchor='top', y=-0.16, xanchor='center', x=0.5,
            font=dict(size=10, color=MUTED), bgcolor='rgba(0,0,0,0)',
        ),
        hoverlabel=dict(bgcolor='white', font_size=11, font_family='Inter, sans-serif',
                        bordercolor='#E5E7EB'),
    )


# =============================================================================
# Section I — State of the Business
# =============================================================================
def chart_mrr_over_time(mrr):
    monthly = mrr.groupby('mrr_month')['mrr'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['mrr_month'], y=monthly['mrr'],
        mode='lines', name='MRR',
        line=dict(color=TEAMBLUE, width=2),
        fill='tozeroy', fillcolor='rgba(30,90,185,0.10)',
    ))
    fig.update_layout(**base_layout(title='Monthly Recurring Revenue (MRR)',
                                    ytitle='€ MRR', height=320, legend=False))
    fig.update_yaxes(tickprefix='€', separatethousands=True)
    return fig


def chart_arpu_over_time(mrr):
    monthly = mrr.groupby('mrr_month').agg(mrr=('mrr', 'sum'),
                                           customers=('customer_id', 'nunique')).reset_index()
    monthly['arpu'] = monthly['mrr'] / monthly['customers']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['mrr_month'], y=monthly['arpu'],
        mode='lines', line=dict(color=TEAMBLUE, width=2),
        fill='tozeroy', fillcolor='rgba(30,90,185,0.08)',
    ))
    fig.update_layout(**base_layout(title='Average Revenue Per User (ARPU)',
                                    ytitle='€ / customer / month', height=340, legend=False))
    fig.update_yaxes(tickprefix='€')
    return fig


def chart_nrr_over_time(mrr):
    """Standard NRR: month-over-month change in existing-customer MRR."""
    monthly = mrr.groupby('mrr_month').agg(
        mrr=('mrr', 'sum'),
        customers=('customer_id', lambda x: set(x))
    ).reset_index()

    nrr = []
    months = []
    for i in range(1, len(monthly)):
        prev_customers = monthly.iloc[i-1]['customers']
        prev_mrr = monthly.iloc[i-1]['mrr']
        # MRR this month from customers who existed last month
        this_mrr_existing = mrr[
            (mrr['mrr_month'] == monthly.iloc[i]['mrr_month']) &
            (mrr['customer_id'].isin(prev_customers))
        ]['mrr'].sum()
        if prev_mrr > 0:
            nrr.append((this_mrr_existing / prev_mrr) * 100)
            months.append(monthly.iloc[i]['mrr_month'])

    nrr_series = pd.Series(nrr, index=months)
    rolling = nrr_series.rolling(3, min_periods=1).mean()

    NRR_COLOR = '#E67E22'  # amber/orange — matches the notebook's NRR chart
    fig = go.Figure()
    fig.add_hline(y=100, line_dash='dash', line_color=MUTED, opacity=0.5)
    fig.add_trace(go.Scatter(x=nrr_series.index, y=nrr_series.values,
                             mode='lines', name='Monthly NRR',
                             line=dict(color=NRR_COLOR, width=1, dash='dot'),
                             opacity=0.5))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values,
                             mode='lines', name='3-month rolling avg',
                             line=dict(color=NRR_COLOR, width=2.5)))
    fig.update_layout(**base_layout(title='Standard Net Revenue Retention (NRR)',
                                    ytitle='%', height=340))
    fig.update_yaxes(ticksuffix='%')
    return fig


def chart_retention_curve(retention_df, title='Revenue Retention by Acquisition Cohort'):
    fig = go.Figure()
    fig.add_hline(y=100, line_dash='dash', line_color=MUTED, opacity=0.4)
    if len(retention_df) == 0:
        fig.update_layout(**base_layout(title=title, height=380))
        return fig
    for cohort in sorted(retention_df['cohort_year'].unique()):
        if int(cohort) >= 2026:
            continue  # 2026 cohort too immature to plot
        sub = retention_df[retention_df['cohort_year'] == cohort].sort_values('months_since')
        sub = sub[sub['months_since'] <= 36]
        fig.add_trace(go.Scatter(
            x=sub['months_since'], y=sub['pct_retained'],
            mode='lines', name=f'Cohort {int(cohort)}',
            line=dict(color=COHORT_PALETTE.get(int(cohort), '#888'), width=2),
        ))
    fig.update_layout(**base_layout(title=title, ytitle='% MRR Retained',
                                    xtitle='Months since acquisition', height=420))
    fig.update_yaxes(ticksuffix='%')
    return fig


def chart_logo_retention_curve(retention_df, title='Customer Retention by Acquisition Cohort'):
    """Logo retention — % of customers in each cohort still active over time.
    Bounded at 100%. Each customer counts as 1, regardless of spend."""
    fig = go.Figure()
    if len(retention_df) == 0:
        fig.update_layout(**base_layout(title=title, height=420))
        return fig
    fig.add_hline(y=100, line_dash='dash', line_color=MUTED, opacity=0.4)
    for cohort in sorted(retention_df['cohort_year'].unique()):
        if int(cohort) >= 2026:
            continue
        sub = retention_df[retention_df['cohort_year'] == cohort].sort_values('months_since')
        sub = sub[sub['months_since'] <= 36]
        fig.add_trace(go.Scatter(
            x=sub['months_since'], y=sub['pct_retained'],
            mode='lines', name=f'Cohort {int(cohort)}',
            line=dict(color=COHORT_PALETTE.get(int(cohort), '#888'), width=2),
        ))
    fig.update_layout(**base_layout(title=title, ytitle='% Customers Retained',
                                    xtitle='Months since acquisition', height=420))
    fig.update_yaxes(ticksuffix='%', range=[0, 105])  # bounded at 100% + tiny headroom
    return fig


# =============================================================================
# Section II–V — MRR by dimension
# =============================================================================
def chart_mrr_by_dimension(mrr, dimension, title):
    monthly = mrr.groupby(['mrr_month', dimension])['mrr'].sum().reset_index()
    fig = go.Figure()
    cats = sorted(monthly[dimension].dropna().unique(), key=lambda x: str(x))
    for i, cat in enumerate(cats):
        sub = monthly[monthly[dimension] == cat]
        fig.add_trace(go.Scatter(
            x=sub['mrr_month'], y=sub['mrr'],
            mode='lines', name=str(cat),
            line=dict(color=CAT_PALETTE[i % len(CAT_PALETTE)], width=2),
        ))
    layout = base_layout(title=title, ytitle='€ MRR', height=400)
    # Many categories → put legend on the right side instead of below
    if len(cats) >= 6:
        layout['legend'] = dict(
            orientation='v', yanchor='top', y=1, xanchor='left', x=1.02,
            font=dict(size=10, color=MUTED), bgcolor='rgba(0,0,0,0)',
        )
        layout['margin'] = dict(l=55, r=160, t=45, b=40)
    fig.update_layout(**layout)
    fig.update_yaxes(tickprefix='€', separatethousands=True)
    return fig


# =============================================================================
# Sankey — where customers go at renewal
# =============================================================================
def chart_renewal_sankey(invoices):
    """First contract subscription_type -> second contract (or 'Lost')."""
    if len(invoices) == 0:
        return go.Figure()

    if 'subscription_type' in invoices.columns:
        inv = invoices.copy()
    else:
        bucket_map = {1: '1 months', 12: '12 months', 24: '24 months',
                      60: '60 months', 120: '120 months'}
        inv = invoices.copy()
        inv['subscription_type'] = (inv['contract_length_months']
                                    .map(bucket_map).fillna('Other'))

    inv_sorted = inv.sort_values(['customer_id', 'invoice_date'])
    first = inv_sorted.groupby('customer_id').first().reset_index()
    second = inv_sorted.groupby('customer_id').nth(1).reset_index()

    first_map = dict(zip(first['customer_id'], first['subscription_type'].astype(str)))
    second_map = dict(zip(second['customer_id'], second['subscription_type'].astype(str)))

    flows = {}
    for cid, first_type in first_map.items():
        if first_type in ['60 months', '120 months']:
            continue  # haven't reached natural renewal yet
        second_type = second_map.get(cid)
        key = (f'{first_type} (start)',
               f'{second_type}' if second_type else 'Lost')
        flows[key] = flows.get(key, 0) + 1

    # Canonical ordering for source / target nodes
    source_order = ['12 months (start)', '24 months (start)',
                    '1 months (start)', 'Other (start)']
    target_order = ['12 months', '24 months', '60 months',
                    '120 months', '1 months', 'Other', 'Lost']

    # Per-contract-type colors so flows are distinguishable by source
    SUB_COLORS = {
        '1 months':   '#d62728',  # red
        '12 months':  '#1f77b4',  # blue
        '24 months':  '#ff7f0e',  # orange
        '60 months':  '#2ca02c',  # green
        '120 months': '#9467bd',  # purple
        'Other':      '#8c564b',  # brown
        'Lost':       '#7f7f7f',  # gray
    }

    def _label_base(label):
        return label.replace(' (start)', '')

    def _with_alpha(hex_color, alpha=0.35):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    label_to_idx = {}
    x_positions = []
    node_colors = []

    for label in source_order:
        if any(s == label for s, _ in flows.keys()):
            label_to_idx[label] = len(label_to_idx)
            x_positions.append(0.001)
            node_colors.append(SUB_COLORS.get(_label_base(label), '#888'))

    for label in target_order:
        if any(t == label for _, t in flows.keys()):
            label_to_idx[label] = len(label_to_idx)
            x_positions.append(0.999)
            node_colors.append(SUB_COLORS.get(label, '#888'))

    sources, targets, values = [], [], []
    for (src, tgt), v in flows.items():
        sources.append(label_to_idx[src])
        targets.append(label_to_idx[tgt])
        values.append(v)

    labels = list(label_to_idx.keys())

    # Link colors: source-colored, but gray when flowing to "Lost" (matches notebook)
    link_colors = []
    for s, t in zip(sources, targets):
        if labels[t] == 'Lost':
            link_colors.append('rgba(127,127,127,0.30)')
        else:
            link_colors.append(_with_alpha(node_colors[s]))

    counts = {}
    for label in labels:
        if 'start' in label:
            counts[label] = sum(v for (s, _), v in flows.items() if s == label)
        else:
            counts[label] = sum(v for (_, t), v in flows.items() if t == label)
    labels_with_counts = [f'{l}  {counts[l]:,}' for l in labels]

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            label=labels_with_counts,
            pad=28, thickness=24,
            line=dict(color='#E5E7EB', width=0.5),
            color=node_colors,
            x=x_positions,
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
        ),
        textfont=dict(family='Inter, sans-serif', size=13, color=INK),
    )])
    fig.update_layout(
        height=580,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family='Inter, sans-serif', size=13, color=INK),
        paper_bgcolor='white',
    )
    return fig


# =============================================================================
# MRR movement waterfall
# =============================================================================
def chart_mrr_movement(mrr):
    """New / Expansion / Contraction / Churned MRR per month."""
    if len(mrr) == 0:
        return go.Figure()
    monthly = mrr.pivot_table(index='mrr_month', columns='customer_id',
                              values='mrr', aggfunc='sum', fill_value=0)
    if len(monthly) < 2:
        return go.Figure()

    new_mrr, expansion, contraction, churned = [], [], [], []
    months = []
    for i in range(1, len(monthly)):
        prev = monthly.iloc[i-1]
        curr = monthly.iloc[i]
        new = curr[(prev == 0) & (curr > 0)].sum()
        ch = -prev[(prev > 0) & (curr == 0)].sum()
        diff = curr - prev
        existing = (prev > 0) & (curr > 0)
        exp = diff[existing & (diff > 0)].sum()
        con = diff[existing & (diff < 0)].sum()
        new_mrr.append(new)
        expansion.append(exp)
        contraction.append(con)
        churned.append(ch)
        months.append(monthly.index[i])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=new_mrr, name='New / Reactivated', marker_color=GREEN))
    fig.add_trace(go.Bar(x=months, y=expansion, name='Expansion', marker_color=TEAMBLUE))
    fig.add_trace(go.Bar(x=months, y=contraction, name='Contraction', marker_color='#F59E0B'))
    fig.add_trace(go.Bar(x=months, y=churned, name='Churned', marker_color=RED))
    fig.update_layout(**base_layout(title='MRR Movement Components',
                                    ytitle='€ MRR change', height=380),
                      barmode='relative')
    fig.update_yaxes(tickprefix='€', separatethousands=True)
    return fig
