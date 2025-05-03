#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import global_utils as gu
import os
from datetime import timedelta

# Constants
N_ITEMS = 5000
N_NOISE_ITEMS = 400
N_CLASSES = 200
N_DEPTS = 20
START_DATE = pd.to_datetime("today").normalize() - pd.DateOffset(years=4)
END_DATE = pd.to_datetime("today").normalize()
DATES = pd.date_range(start=START_DATE, end=END_DATE, freq='W-SAT')

logger = gu.setup_logging("gen_sales")


def generate_noise_items(n, classes, depts):
    logger.info(f"Generating {n} noise items")
    noise_items = []
    for _ in range(n):
        season = np.random.choice(['BASIC', 'SPRING', 'FALL'])
        item_properties = define_item_seasonal_properties(season)
        item = {
            'Item': f'NOISE{np.random.randint(20, 100)}',
            'Class': np.random.choice(classes),
            'Dept': np.random.choice(depts),
            'Season': season,
            'Properties': item_properties
        }
        noise_items.append(item)
    return noise_items


def define_item_seasonal_properties(season):
    logger.debug(f"Defining seasonal properties for {season}")
    if season == "SPRING":
        start_week = np.random.randint(9, 21)
    elif season == "FALL":
        start_week = np.random.randint(40, 52) if np.random.rand() > 0.5 else np.random.randint(0, 5)
    else:
        start_week = np.random.randint(0, 52)

    return {
        'start_week': start_week,
        'ramp_up_duration': np.random.randint(1, 3),
        'peak_duration': np.random.randint(8, 13),
        'sell_down_duration': np.random.randint(18, 22)
    }


def apply_trend(sales, trend_percent, total_weeks):
    logger.debug(f"Applying trend of {trend_percent}% over {total_weeks} weeks")
    if total_weeks > 1:
        trend_factors = 1 + (trend_percent / 100) * (np.arange(total_weeks) / (total_weeks - 1))
        return sales * trend_factors
    return sales


def generate_sales(item, dates, year_labels):
    logger.debug(f"Generating sales for item {item['Item']}")
    total_weeks = len(dates)
    sales = np.zeros(total_weeks, dtype=float)

    # Create active period mask
    active_mask = (year_labels >= item['start_year']) & (year_labels <= item['end_year'])
    active_weeks = np.where(active_mask)[0]

    if len(active_weeks) == 0:
        return sales.round().astype(int)

    if item['Season'] in ['SPRING', 'FALL']:
        weeks_per_year = total_weeks // 4
        for year in range(4):
            if not active_mask[year * weeks_per_year]:
                continue

            year_start = year * weeks_per_year
            year_end = year_start + weeks_per_year

            start_week = year_start + item['Properties']['start_week']
            intro_duration = item['Properties']['ramp_up_duration']
            growth_duration = item['Properties']['peak_duration']
            decline_duration = item['Properties']['sell_down_duration']

            base_sales = np.random.randint(50, 150)
            peak_sales = np.random.randint(450, 1500)

            intro_growth = np.linspace(base_sales, peak_sales, intro_duration)
            peak_phase = np.full(growth_duration, peak_sales)
            decline_phase = np.linspace(peak_sales, base_sales, decline_duration)

            end_intro = min(start_week + intro_duration, year_end)
            end_peak = min(end_intro + growth_duration, year_end)
            end_decline = min(end_peak + decline_duration, year_end)

            sales[start_week:end_intro] = intro_growth[:end_intro - start_week]
            sales[end_intro:end_peak] = peak_phase[:end_peak - end_intro]
            sales[end_peak:end_decline] = decline_phase[:end_decline - end_peak]

    elif item['Season'] == 'BASIC':
        base_sales = np.random.randint(200, 1000)
        fluctuations = np.random.normal(0, base_sales * 0.1, total_weeks)
        sales[:] = base_sales + fluctuations

    # Apply trend and active period mask
    sales = apply_trend(sales, item['trend'], total_weeks)
    sales *= active_mask.astype(float)

    sales += np.random.normal(0, 50, total_weeks)
    apply_christmas_bump(sales, dates)

    return np.clip(sales, 50, None).round().astype(int)


def apply_christmas_bump(sales, dates):
    logger.debug("Applying Christmas sales bump")
    christmas_period = (dates.month == 12) & ((dates.day > 10) & (dates.day < 26))
    sales[christmas_period] *= np.random.uniform(1.5, 3.0, size=np.sum(christmas_period))


def main():
    logger.info("Starting sales data generation")

    # Generate initial items
    logger.info("Generating base items")
    items = pd.DataFrame()
    year_labels = (DATES.year - START_DATE.year) + 1

    # Initialize items with 4-year lifespan
    base_items = pd.DataFrame({
        'Item': [f'STYCOL{i:06d}' for i in range(1, N_ITEMS + 1)],
        'Class': [f'C{i:06d}' for i in np.random.randint(1, N_CLASSES + 1, N_ITEMS)],
        'Dept': [f'D{i:06d}' for i in np.random.randint(1, N_DEPTS + 1, N_ITEMS)],
        'Season': np.random.choice(['BASIC', 'SPRING', 'FALL'], size=N_ITEMS, p=[0.5, 0.25, 0.25]),
        'start_year': 1,
        'end_year': 4
    })
    items = pd.concat([items, base_items], ignore_index=True)

    # Add initial noise items
    logger.info("Adding initial noise items")
    noise_items = generate_noise_items(N_NOISE_ITEMS, base_items['Class'].unique(), base_items['Dept'].unique())
    noise_df = pd.DataFrame(noise_items)
    noise_df['start_year'] = 1
    noise_df['end_year'] = 4
    items = pd.concat([items, noise_df], ignore_index=True)

    # Annual item turnover
    for year in range(2, 5):
        logger.info(f"Processing year {year} turnover")
        active_items = items[(items['start_year'] <= year) & (items['end_year'] >= year)]
        n_replace = int(len(active_items) * 0.08)

        if n_replace > 0:
            # Retire items
            retired = active_items.sample(n_replace)
            items.loc[retired.index, 'end_year'] = year - 1

            # Generate replacements
            new_items = pd.DataFrame({
                'Item': [f'STYCOL{i:06d}' for i in range(len(items) + 1, len(items) + n_replace + 1)],
                'Class': np.random.choice(items['Class'].unique(), n_replace),
                'Dept': np.random.choice(items['Dept'].unique(), n_replace),
                'Season': np.random.choice(['BASIC', 'SPRING', 'FALL'], size=n_replace, p=[0.5, 0.25, 0.25]),
                'start_year': year,
                'end_year': 4
            })
            items = pd.concat([items, new_items], ignore_index=True)

    # Add seasonal patterns and trends
    logger.info("Finalizing item configurations")
    items['Properties'] = items.apply(lambda x: define_item_seasonal_properties(x['Season']), axis=1)
    items['trend'] = np.random.choice([1, -1], size=len(items), p=[0.8, 0.2]) * np.random.uniform(1, 10,
                                                                                                  size=len(items))

    # Generate sales data
    logger.info("Generating sales records")
    all_sales = []
    for _, row in items.iterrows():
        sales = generate_sales(row, DATES, year_labels)
        item_df = pd.DataFrame({
            'Date': DATES,
            'Item': row['Item'],
            'Sales': sales,
            'Class': row['Class'],
            'Dept': row['Dept'],
            'Season': row['Season'],
        })

        all_sales.append(item_df)

    # Combine and add time features
    logger.info("Compiling final dataset")
    final_df = pd.concat(all_sales, ignore_index=True)
    dt = final_df['Date'].dt
    final_df['woy'] = dt.isocalendar().week.astype('int32')
    final_df['moy'] = dt.month.astype('int32')
    final_df['qoy'] = dt.quarter.astype('int32')
    final_df['soy'] = dt.month.map(lambda x: 1 if x <= 6 else 2).astype('int32')
    final_df['year'] = dt.year.astype('int32')

    # Merge trend data
    final_df = pd.merge(final_df, items[['Item', 'trend']], on='Item', how='left')

    # Save data
    output_path = os.path.join(gu.RAW_DIR, 'generated_sales.csv')
    final_df.to_csv(output_path, index=False)
    logger.info(f"Dataset generated with annual turnover and trend. Saved to {output_path}")
    logger.info(f"Final item count: {len(items)} ({len(items) - N_ITEMS - N_NOISE_ITEMS} replacements added)")


if __name__ == "__main__":
    main()
