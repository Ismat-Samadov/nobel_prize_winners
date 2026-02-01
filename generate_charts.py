"""
Nobel Prize Analysis - Business Insights Generator
Generates executive-level visualizations from Nobel Prize data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for professional charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11

# Create charts directory
charts_dir = Path('charts')
charts_dir.mkdir(exist_ok=True)

# Load data
print("Loading Nobel Prize data...")
df = pd.read_csv('data/SDnobel.csv')

# Clean and prepare data
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['age'] = pd.to_numeric(df['age'], errors='coerce')

print(f"Total records: {len(df)}")
print(f"Years covered: {df['year'].min():.0f} - {df['year'].max():.0f}")
print(f"Categories: {df['category'].nunique()}")

# ============================================================================
# CHART 1: Award Distribution by Category Over Time
# ============================================================================
print("\nGenerating Chart 1: Award trends by category...")

fig, ax = plt.subplots(figsize=(14, 7))
category_by_decade = df.groupby(['decade', 'category']).size().unstack(fill_value=0)
category_by_decade.plot(kind='bar', ax=ax, width=0.8)

ax.set_title('Nobel Prize Awards by Category Across Decades', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Decade', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Awards', fontsize=12, fontweight='bold')
ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(charts_dir / '01_category_distribution_by_decade.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 2: Gender Diversity Evolution
# ============================================================================
print("Generating Chart 2: Gender diversity trends...")

# Get gender breakdown by decade
gender_decade = df[df['laureate_type'] == 'Individual'].groupby(['decade', 'sex']).size().unstack(fill_value=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Absolute numbers
gender_decade.plot(kind='bar', ax=ax1, color=['#FF6B9D', '#4A90E2'], width=0.7)
ax1.set_title('Gender Representation Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Decade', fontsize=12)
ax1.set_ylabel('Number of Laureates', fontsize=12)
ax1.legend(title='Gender')
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Percentage
gender_pct = gender_decade.div(gender_decade.sum(axis=1), axis=0) * 100
gender_pct.plot(kind='bar', ax=ax2, stacked=True, color=['#FF6B9D', '#4A90E2'], width=0.7)
ax2.set_title('Gender Distribution (Percentage)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Decade', fontsize=12)
ax2.set_ylabel('Percentage of Laureates', fontsize=12)
ax2.legend(title='Gender')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 100)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(charts_dir / '02_gender_diversity_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 3: Top 15 Countries by Total Winners
# ============================================================================
print("Generating Chart 3: Geographic concentration...")

# Count by birth country
country_counts = df[df['laureate_type'] == 'Individual']['birth_country'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(country_counts)), country_counts.values, color='#2E86AB')
ax.set_yticks(range(len(country_counts)))
ax.set_yticklabels(country_counts.index)
ax.set_xlabel('Number of Nobel Laureates', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Countries by Nobel Laureates (Birthplace)', fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(country_counts.values):
    ax.text(v + 2, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(charts_dir / '03_top_countries_by_winners.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 4: Top 20 Institutions/Organizations
# ============================================================================
print("Generating Chart 4: Institutional performance...")

# Count by organization
org_counts = df[df['organization_name'].notna()]['organization_name'].value_counts().head(20)

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(range(len(org_counts)), org_counts.values, color='#A23B72')
ax.set_yticks(range(len(org_counts)))
ax.set_yticklabels(org_counts.index, fontsize=10)
ax.set_xlabel('Number of Laureates Affiliated', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Institutions by Nobel Laureates', fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(org_counts.values):
    ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(charts_dir / '04_top_institutions.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 5: Age Distribution at Award Time
# ============================================================================
print("Generating Chart 5: Age demographics...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Age histogram
age_data = df[df['age'].notna()]['age']
ax1.hist(age_data, bins=30, color='#18A558', edgecolor='black', alpha=0.7)
ax1.axvline(age_data.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {age_data.median():.0f} years')
ax1.axvline(age_data.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {age_data.mean():.0f} years')
ax1.set_xlabel('Age at Award', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Laureates', fontsize=12, fontweight='bold')
ax1.set_title('Age Distribution of Nobel Laureates', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Age by category
age_by_cat = df[df['age'].notna()].groupby('category')['age'].median().sort_values()
bars = ax2.barh(range(len(age_by_cat)), age_by_cat.values, color='#F18F01')
ax2.set_yticks(range(len(age_by_cat)))
ax2.set_yticklabels(age_by_cat.index)
ax2.set_xlabel('Median Age at Award', fontsize=12, fontweight='bold')
ax2.set_title('Median Age by Category', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

for i, v in enumerate(age_by_cat.values):
    ax2.text(v + 0.5, i, f'{v:.0f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(charts_dir / '05_age_demographics.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 6: USA vs Rest of World Performance
# ============================================================================
print("Generating Chart 6: USA market share analysis...")

# USA vs non-USA by decade
df['region'] = df['usa_born_winner'].apply(lambda x: 'USA' if x == True else 'Rest of World')
usa_decade = df[df['laureate_type'] == 'Individual'].groupby(['decade', 'region']).size().unstack(fill_value=0)

# Chart 6A: Absolute numbers
fig, ax = plt.subplots(figsize=(14, 8))
usa_decade.plot(kind='bar', ax=ax, color=['#C9ADA7', '#22223B'], width=0.75)
ax.set_title('USA vs Rest of World - Nobel Laureates by Decade', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Decade', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Laureates', fontsize=14, fontweight='bold')
ax.legend(title='Region', fontsize=12, title_fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(charts_dir / '06_usa_vs_world_absolute.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 6B: Market share percentage
fig, ax = plt.subplots(figsize=(14, 8))
usa_pct = usa_decade.div(usa_decade.sum(axis=1), axis=0) * 100
usa_pct['USA'].plot(kind='line', ax=ax, marker='o', linewidth=4, markersize=12, color='#22223B')
ax.set_title('USA Market Share in Nobel Prizes Over Time', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Decade', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage of Total Laureates', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Add percentage labels
for x, y in zip(range(len(usa_pct)), usa_pct['USA'].values):
    ax.text(x, y + 3, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(charts_dir / '06_usa_market_share.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 7: Prize Sharing Trends
# ============================================================================
print("Generating Chart 7: Collaboration patterns...")

# Analyze prize sharing
df['shared'] = df['prize_share'].apply(lambda x: 'Shared' if x != '1/1' else 'Individual')
sharing_decade = df.groupby(['decade', 'shared']).size().unstack(fill_value=0)

# Chart 7A: Absolute numbers
fig, ax = plt.subplots(figsize=(14, 8))
sharing_decade.plot(kind='bar', ax=ax, color=['#06A77D', '#D4A373'], width=0.75)
ax.set_title('Individual vs Shared Nobel Prizes by Decade', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Decade', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Awards', fontsize=14, fontweight='bold')
ax.legend(title='Award Type', fontsize=12, title_fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(charts_dir / '07_collaboration_absolute.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 7B: Percentage of shared prizes
fig, ax = plt.subplots(figsize=(14, 8))
sharing_pct = sharing_decade.div(sharing_decade.sum(axis=1), axis=0) * 100
sharing_pct['Shared'].plot(kind='line', ax=ax, marker='s', linewidth=4, markersize=12, color='#D4A373')
ax.set_title('Collaboration Trend: Percentage of Shared Nobel Prizes', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Decade', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage of Shared Prizes', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Add percentage labels
for x, y in zip(range(len(sharing_pct)), sharing_pct['Shared'].values):
    ax.text(x, y + 3, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(charts_dir / '07_collaboration_percentage.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 8: Category-wise Country Dominance
# ============================================================================
print("Generating Chart 8: Category leadership by country...")

# Top 5 countries per category
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

categories = df['category'].unique()
for idx, category in enumerate(sorted(categories)):
    if idx < 6:
        cat_data = df[(df['category'] == category) & (df['laureate_type'] == 'Individual')]
        top_countries = cat_data['birth_country'].value_counts().head(10)

        axes[idx].barh(range(len(top_countries)), top_countries.values,
                      color=plt.cm.Set3(range(len(top_countries))))
        axes[idx].set_yticks(range(len(top_countries)))
        axes[idx].set_yticklabels(top_countries.index, fontsize=9)
        axes[idx].set_xlabel('Number of Laureates', fontsize=10)
        axes[idx].set_title(f'{category}', fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(top_countries.values):
            axes[idx].text(v + 0.5, i, str(v), va='center', fontsize=9, fontweight='bold')

plt.suptitle('Top 10 Countries by Nobel Category', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(charts_dir / '08_category_country_leadership.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 9: Decade Performance Summary
# ============================================================================
print("Generating Chart 9: Decade-based performance...")

decade_summary = df.groupby('decade').agg({
    'laureate_id': 'count',
    'female': lambda x: (x == True).sum(),
    'usa_born_winner': lambda x: (x == True).sum()
}).rename(columns={
    'laureate_id': 'Total Awards',
    'female': 'Female Winners',
    'usa_born_winner': 'USA Winners'
})

fig, ax = plt.subplots(figsize=(14, 7))
decade_summary.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Nobel Prize Metrics by Decade', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Decade', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Awards/Winners', fontsize=12, fontweight='bold')
ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(charts_dir / '09_decade_performance_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CHART 10: Growth Trends Over Time
# ============================================================================
print("Generating Chart 10: Overall growth trends...")

yearly_counts = df.groupby('year').size()
cumulative_counts = yearly_counts.cumsum()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Annual awards
ax1.plot(yearly_counts.index, yearly_counts.values, linewidth=2, color='#1982C4', marker='o', markersize=3)
ax1.set_title('Annual Nobel Prize Awards', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of Awards', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(yearly_counts.mean(), color='red', linestyle='--',
           label=f'Average: {yearly_counts.mean():.1f} per year', linewidth=2)
ax1.legend()

# Cumulative awards
ax2.fill_between(cumulative_counts.index, cumulative_counts.values, alpha=0.3, color='#6A4C93')
ax2.plot(cumulative_counts.index, cumulative_counts.values, linewidth=3, color='#6A4C93')
ax2.set_title('Cumulative Nobel Prize Awards Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Cumulative Number of Awards', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(charts_dir / '10_growth_trends_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Generate Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS FOR BUSINESS INSIGHTS")
print("="*70)

print(f"\nTotal Nobel Prizes Awarded: {len(df)}")
print(f"Individual Laureates: {len(df[df['laureate_type'] == 'Individual'])}")
print(f"Organizations: {len(df[df['laureate_type'] == 'Organization'])}")

print(f"\nGender Distribution:")
gender_dist = df[df['laureate_type'] == 'Individual']['sex'].value_counts()
for gender, count in gender_dist.items():
    pct = (count / gender_dist.sum()) * 100
    print(f"  {gender}: {count} ({pct:.1f}%)")

print(f"\nTop 5 Countries (by birth):")
for country, count in df[df['laureate_type'] == 'Individual']['birth_country'].value_counts().head(5).items():
    print(f"  {country}: {count}")

print(f"\nTop 5 Organizations:")
for org, count in df[df['organization_name'].notna()]['organization_name'].value_counts().head(5).items():
    print(f"  {org}: {count}")

print(f"\nAge Statistics:")
print(f"  Average age at award: {df['age'].mean():.1f} years")
print(f"  Median age at award: {df['age'].median():.1f} years")
print(f"  Youngest laureate: {df['age'].min():.0f} years")
print(f"  Oldest laureate: {df['age'].max():.0f} years")

print(f"\nCategory Distribution:")
for cat, count in df['category'].value_counts().items():
    print(f"  {cat}: {count}")

print(f"\nUSA Performance:")
usa_count = len(df[df['usa_born_winner'] == True])
total_individual = len(df[df['laureate_type'] == 'Individual'])
print(f"  USA-born winners: {usa_count} ({(usa_count/total_individual)*100:.1f}% of individuals)")

print(f"\nCollaboration Trends:")
shared_count = len(df[df['prize_share'] != '1/1'])
print(f"  Shared prizes: {shared_count} ({(shared_count/len(df))*100:.1f}%)")

print("\n" + "="*70)
print("All charts generated successfully in 'charts/' directory!")
print("="*70)
