import matplotlib.pyplot as plt
from A4_datawrangling import *
import os
os.environ["OMP_NUM_THREADS"] = "1"


X, y = get_dataset(outlier_removal='none')
df_plot = pd.concat([X, y], axis=1)

def plot_season_by_likelihood(df, level, output_dir="descriptive"):

    subset = df[df_plot['Mareks_Diagnosis_Level'] == level]

    # season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    region_order = ['East Midlands',
'East of England',
'London',
'North East',
'North West',
'Scotland',
'South East',
'South West',
'Wales',
'West Midlands',
'Yorkshire and The Humber'
]
    counts = subset['Region'].value_counts().reindex(region_order, fill_value=0)

    plt.figure(figsize=(6,4))
    counts.plot(kind='bar')
    plt.xlabel('Region')
    plt.ylabel(f'Count of "{level}" Records')
    plt.title(f'{level} Likelihood of Marek’s Disease by Region')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    os.makedirs(output_dir, exist_ok=True)

    filename   = "{safe_level}.png"  
    filepath   = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

for lvl in ['Very Low', 'Low', 'High', 'Very High']:
    plot_season_by_likelihood(df, lvl)

print(df["Mareks_Diagnosis_Level"].value_counts())