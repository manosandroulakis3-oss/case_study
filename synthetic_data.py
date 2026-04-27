"""
Synthetic data generator. Produces dataframes with the same schema as
the team.blue notebook so the app runs immediately for design/preview.

Replace this with real data loading in `data_prep.py` when ready.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COUNTRIES = ['Italy', 'Netherlands', 'Spain', 'Portugal', 'Belgium',
             'Hungary', 'Sweden', 'Czechia', 'UK', 'France', 'Finland', 'Poland']
COUNTRY_WEIGHTS = [0.16, 0.18, 0.12, 0.08, 0.10, 0.07, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03]

PRODUCT_GROUPS = ['Shared hosting', '.com', 'Other TLDs', 'Local ccTLD',
                  'WHOIS Privacy', 'Compliance', 'Other Domain Services']
GROUP_WEIGHTS = [0.50, 0.16, 0.10, 0.10, 0.11, 0.02, 0.01]

SUBGROUPS = {
    'Shared hosting': ['Plans', 'Add-on'],
    '.com': ['.com'],
    'Other TLDs': ['Other TLDs'],
    'Local ccTLD': ['Local ccTLD'],
    'WHOIS Privacy': ['WHOIS Privacy'],
    'Compliance': ['Compliance'],
    'Other Domain Services': ['Other Domain Services'],
}

CONTRACT_LENGTHS = [1, 12, 24, 60, 120]
CONTRACT_WEIGHTS = [0.003, 0.79, 0.18, 0.025, 0.002]

PLATFORM_DEPTH_LABELS = {1: '1 - Single Product', 2: '2 - Two Products', 3: '3 - Multi Product (3+)'}


def generate_synthetic_data(n_customers=70_938):
    """Generate customers, invoices, and exploded MRR dataframes."""

    # === CUSTOMERS ===
    cohort_year_weights = [0.27, 0.22, 0.18, 0.17, 0.13, 0.03]  # 2021..2026
    cohort_years = RNG.choice([2021, 2022, 2023, 2024, 2025, 2026],
                              size=n_customers, p=cohort_year_weights)

    cohort_months = []
    for y in cohort_years:
        if y == 2026:
            m = RNG.integers(1, 5)  # up to April
        else:
            m = RNG.integers(1, 13)
        cohort_months.append(pd.Timestamp(year=int(y), month=int(m), day=1))

    customers = pd.DataFrame({
        'customer_id': np.arange(1, n_customers + 1),
        'country': RNG.choice(COUNTRIES, size=n_customers, p=COUNTRY_WEIGHTS),
        'cohort_month': cohort_months,
        'cohort_year': cohort_years,
    })

    # Platform depth: assigned probabilistically, with later cohorts more likely to be single product
    depth_probs = {
        2021: [0.35, 0.30, 0.35],
        2022: [0.40, 0.30, 0.30],
        2023: [0.45, 0.28, 0.27],
        2024: [0.50, 0.27, 0.23],
        2025: [0.55, 0.25, 0.20],
        2026: [0.60, 0.25, 0.15],
    }
    depths = []
    for y in customers['cohort_year']:
        depths.append(RNG.choice([1, 2, 3], p=depth_probs[y]))
    customers['platform_depth_n'] = depths
    customers['platform_depth'] = customers['platform_depth_n'].map(PLATFORM_DEPTH_LABELS)

    # === INVOICES ===
    invoices = []
    for _, row in customers.iterrows():
        depth = row['platform_depth_n']
        n_products = depth if depth < 3 else RNG.integers(3, 6)
        # First contract — pick contract length
        contract_len = RNG.choice(CONTRACT_LENGTHS, p=CONTRACT_WEIGHTS)

        for prod_idx in range(n_products):
            group = RNG.choice(PRODUCT_GROUPS, p=GROUP_WEIGHTS)
            subgroup = RNG.choice(SUBGROUPS[group])

            # Per-product per-month price band (roughly matching report figures)
            arpu_band = {
                'Shared hosting': (4, 12),
                '.com': (1.0, 2.5),
                'Other TLDs': (1.0, 2.5),
                'Local ccTLD': (1.0, 2.5),
                'WHOIS Privacy': (0.4, 1.0),
                'Compliance': (1.0, 3.0),
                'Other Domain Services': (5, 30),
            }[group]
            monthly_price = RNG.uniform(*arpu_band)
            invoice_amount = round(monthly_price * contract_len, 2)

            invoices.append({
                'customer_id': row['customer_id'],
                'invoice_date': row['cohort_month'] + pd.DateOffset(days=int(RNG.integers(0, 28))),
                'amount': invoice_amount,
                'contract_length_months': contract_len,
                'product_group': group,
                'product_subgroup': subgroup,
            })

            # Renewal logic — probabilistic second contract
            renewal_prob = {1: 0.10, 12: 0.27, 24: 0.19, 60: 0.03, 120: 0.02}[contract_len]
            renewal_prob *= {1: 0.5, 2: 1.2, 3: 1.7}[depth]  # multi-product renews more
            if RNG.random() < min(renewal_prob, 0.95):
                renewal_date = row['cohort_month'] + pd.DateOffset(months=int(contract_len))
                if renewal_date < pd.Timestamp('2026-04-01'):
                    invoices.append({
                        'customer_id': row['customer_id'],
                        'invoice_date': renewal_date,
                        'amount': invoice_amount * RNG.uniform(0.9, 1.15),
                        'contract_length_months': contract_len,
                        'product_group': group,
                        'product_subgroup': subgroup,
                    })

    inv_df = pd.DataFrame(invoices)
    inv_df = inv_df.merge(customers[['customer_id', 'country', 'cohort_year', 'cohort_month',
                                     'platform_depth', 'platform_depth_n']], on='customer_id')

    # === EXPLODED MRR ===
    # Spread each invoice evenly across its active months
    mrr_rows = []
    for _, inv in inv_df.iterrows():
        n_months = inv['contract_length_months']
        monthly = inv['amount'] / n_months
        start = pd.Timestamp(inv['invoice_date']).to_period('M').to_timestamp()
        for k in range(n_months):
            mrr_month = start + pd.DateOffset(months=k)
            if mrr_month >= pd.Timestamp('2026-05-01'):
                break
            mrr_rows.append({
                'customer_id': inv['customer_id'],
                'mrr_month': mrr_month,
                'mrr': monthly,
                'country': inv['country'],
                'cohort_year': inv['cohort_year'],
                'cohort_month': inv['cohort_month'],
                'platform_depth': inv['platform_depth'],
                'platform_depth_n': inv['platform_depth_n'],
                'product_group': inv['product_group'],
                'product_subgroup': inv['product_subgroup'],
                'contract_length_months': inv['contract_length_months'],
            })

    mrr_df = pd.DataFrame(mrr_rows)

    return customers, inv_df, mrr_df


if __name__ == '__main__':
    import time
    t = time.time()
    c, i, m = generate_synthetic_data(n_customers=2000)  # small for testing
    print(f"Generated in {time.time()-t:.1f}s")
    print(f"Customers: {len(c):,}")
    print(f"Invoices:  {len(i):,}")
    print(f"MRR rows:  {len(m):,}")
