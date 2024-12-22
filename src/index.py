import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Check if the file exists
file_path = '../data/raw/Renewable_Energy_Usage_Sampled.csv'
if os.path.exists(file_path):
    # Saving the cleaned data for further analysis
    data = pd.read_csv(file_path)
    data.to_csv('../data/processed/cleaned_data.csv', index=False)


    # Global Trends Analysis

    global_trends = data.groupby('Year')['Monthly_Usage_kWh'].sum().reset_index()
    global_trends.columns = ['Year', 'Total_Energy_Usage_kWh']
    print(global_trends)

    # plt.figure(figsize=(10, 6))
    # plt.plot(global_trends['Year'], global_trends['Total_Energy_Usage_kWh'], marker='o', linestyle='-', color='b')
    # plt.title('Global Trends of Renewable Energy Usage (2020-2024)', fontsize=16)
    # plt.xlabel('Year', fontsize=14)
    # plt.ylabel('Total Energy Usage (kWh)', fontsize=14)
    # plt.grid(True)
    # plt.xticks(global_trends['Year'])
    # plt.tight_layout()
    

    # os.makedirs('../figures', exist_ok=True)
    # plt.savefig('../figures/global_renewable_energy_usage.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # Regional Energy Usage Analysis
    region_usage = data.groupby('Region')['Monthly_Usage_kWh'].sum().reset_index()
    region_usage.columns = ['Region', 'Total_Energy_Usage_kWh']
    region_usage = region_usage.sort_values(by='Total_Energy_Usage_kWh', ascending=False)

    print(region_usage)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=region_usage, x='Region', y='Total_Energy_Usage_kWh', palette='viridis')
    plt.title('Energy Usage by Region (2020-2024)', fontsize=16)
    plt.xlabel('Region', fontsize=14)
    plt.ylabel('Total Energy Usage (kWh)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../figures/energy_usage_by_region.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print(f"File not found at: {file_path}")