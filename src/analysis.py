import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(file_path):
    """
    讀取資料檔案
    
    Args:
        file_path (str): CSV 檔案路徑
        
    Returns:
        pandas.DataFrame: 讀取的資料框架
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案：{file_path}")
    return pd.read_csv(file_path)

def sort_region_data(df):
    """
    整理地區資料，計算各地區、年份和能源類型的使用量
    
    Args:
        df (pandas.DataFrame): 原始資料框架
        
    Returns:
        tuple: (region_year_source, regions, years, energy_sources, data)
    """
    # 聚合數據：按地區、年份和能源類型計算總使用量
    region_year_source = df.groupby(['Region', 'Year', 'Energy_Source'])['Monthly_Usage_kWh'].sum().reset_index()
    
    # 獲取唯一值列表
    regions = region_year_source['Region'].unique()
    years = region_year_source['Year'].unique()
    energy_sources = region_year_source['Energy_Source'].unique()
    
    # 准備堆疊柱狀圖數據
    data = {}
    for source in energy_sources:
        source_data = region_year_source[region_year_source['Energy_Source'] == source]
        pivot_data = source_data.pivot(index='Region', columns='Year', values='Monthly_Usage_kWh').fillna(0)
        data[source] = pivot_data
    
    return region_year_source, regions, years, energy_sources, data

def plot_region_data(region_year_source, regions, years, energy_sources, data, save_path=None, save_format='png', dpi=300):
    """
    繪製地區能源使用量的堆疊柱狀圖並儲存
    
    Args:
        region_year_source (pandas.DataFrame): 按地區、年份和能源類型聚合的資料
        regions (numpy.array): 唯一地區列表
        years (numpy.array): 唯一年份列表
        energy_sources (numpy.array): 唯一能源類型列表
        data (dict): 依能源類型分類的樞紐表資料
        save_path (str, optional): 圖表儲存路徑。若未指定則只顯示不儲存
        save_format (str, optional): 圖片格式，預設為 'png'。支援 'png', 'jpg', 'pdf', 'svg' 等
        dpi (int, optional): 圖片解析度，預設為 300
    """
    bar_width = 0.12  # 每年度柱狀體寬度
    year_spacing = 0.05  # 每個年份柱狀體之間的額外間隔
    group_spacing = 0.15  # 每個地區之間的額外間隔
    x = np.arange(len(regions)) * (len(years) * (bar_width + year_spacing) + group_spacing)  # X 軸位置

    plt.figure(figsize=(16, 8))

    for i, year in enumerate(years):
        bottom = np.zeros(len(regions))
        for source in energy_sources:
            usage = data[source][year].values if year in data[source].columns else np.zeros(len(regions))
            plt.bar(
                x + i * (bar_width + year_spacing),  # 每年度的柱狀體位置
                usage,  # 當前能源類型的數據
                bar_width,  # 柱狀體寬度
                bottom=bottom,  # 堆疊底部
                label=f'{source}' if year == years[0] else None  # 僅為第一年度標註能源類型
            )
            bottom += usage  # 更新堆疊底部

    # 設置 X 軸和圖表細節
    plt.xticks(x + (len(years) - 1) * bar_width / 2, regions, rotation=45, fontsize=10)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Total Monthly Usage (kWh)', fontsize=12)
    plt.title('Renewable Energy Usage by Region, Year, and Source (2020–2024)', fontsize=16)
    plt.legend(title='Energy Source', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()

    # 儲存圖表
    if save_path:
        # 確保儲存路徑存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 如果檔名沒有副檔名，加上指定的格式
        if not os.path.splitext(save_path)[1]:
            save_path = f"{save_path}.{save_format}"
            
        try:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches='tight',
                format=save_format
            )
            print(f"圖表已儲存至：{save_path}")
        except Exception as e:
            print(f"儲存圖表時發生錯誤：{str(e)}")
    
    # 顯示圖表
    plt.show()
    
    # 清除當前圖表（避免記憶體洩漏）
    plt.close()


def calculate_region_summary(region_year_source):
    """
    計算地區資料的統計摘要
    
    Args:
        region_year_source (pandas.DataFrame): 按地區、年份和能源類型聚合的資料
        
    Returns:
        pandas.DataFrame: 包含總使用量、年均增長率、主要能源類型等統計資料
    """
    # 總使用量
    total_usage = region_year_source.groupby('Region')['Monthly_Usage_kWh'].sum()

    # 年均增長率
    yearly_usage = region_year_source.groupby(['Region', 'Year'])['Monthly_Usage_kWh'].sum()
    yearly_usage = yearly_usage.reset_index()
    yearly_usage_pivot = yearly_usage.pivot(index='Region', columns='Year', values='Monthly_Usage_kWh')
    growth_rates = yearly_usage_pivot.pct_change(axis=1)
    average_growth = growth_rates.mean(axis=1)

    # 最常用的能源類型
    dominant_energy = region_year_source.groupby(['Region', 'Energy_Source'])['Monthly_Usage_kWh'].sum().reset_index()
    dominant_energy = dominant_energy.loc[dominant_energy.groupby('Region')['Monthly_Usage_kWh'].idxmax()]

    # 合併結果
    summary = pd.DataFrame({
        'Total_Usage_kWh': total_usage,
        'Average_Growth_Rate': average_growth,
        'Dominant_Energy_Source': dominant_energy['Energy_Source'].values,
        'Dominant_Energy_Percentage': dominant_energy['Monthly_Usage_kWh'].values / total_usage.values
    })
    
    return summary

def plot_total_usage(summary, save_path=None, save_format='png', dpi=300):
    """
    繪製各地區總使用量的水平條形圖
    
    Args:
        summary (pandas.DataFrame): 地區統計摘要資料
        save_path (str, optional): 圖表儲存路徑
        save_format (str, optional): 圖片格式，預設為 'png'
        dpi (int, optional): 圖片解析度，預設為 300
    """
    plt.figure(figsize=(12, 6))
    
    # 排序數據
    sorted_data = summary['Total_Usage_kWh'].sort_values(ascending=True)
    
    # 繪製水平條形圖
    plt.barh(range(len(sorted_data)), sorted_data.values)
    plt.yticks(range(len(sorted_data)), sorted_data.index)
    
    plt.title('Total Energy Usage by Region', fontsize=14)
    plt.xlabel('Total Usage (kWh)', fontsize=12)
    plt.ylabel('Region', fontsize=12)
    
    plt.tight_layout()
    
    # 儲存圖表
    save_figure(plt, save_path, save_format, dpi)

def plot_growth_rates(summary, save_path=None, save_format='png', dpi=300):
    """
    繪製各地區年度增長率的折線圖
    
    Args:
        summary (pandas.DataFrame): 地區統計摘要資料
        save_path (str, optional): 圖表儲存路徑
        save_format (str, optional): 圖片格式，預設為 'png'
        dpi (int, optional): 圖片解析度，預設為 300
    """
    plt.figure(figsize=(12, 6))
    
    # 排序數據
    sorted_data = summary['Average_Growth_Rate'].sort_values(ascending=False)
    
    # 繪製折線圖
    plt.plot(range(len(sorted_data)), sorted_data.values, 'o-')
    plt.xticks(range(len(sorted_data)), sorted_data.index, rotation=45)
    
    plt.title('Average Growth Rate by Region', fontsize=14)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Average Growth Rate (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 儲存圖表
    save_figure(plt, save_path, save_format, dpi)

def plot_energy_composition(region_year_source, save_path=None, save_format='png', dpi=300):
    """
    繪製各地區能源佔比的堆疊柱狀圖
    
    Args:
        region_year_source (pandas.DataFrame): 按地區、年份和能源類型聚合的資料
        save_path (str, optional): 圖表儲存路徑
        save_format (str, optional): 圖片格式，預設為 'png'
        dpi (int, optional): 圖片解析度，預設為 300
    """
    # 計算各地區的能源佔比
    total_by_region = region_year_source.groupby(['Region', 'Energy_Source'])['Monthly_Usage_kWh'].sum().unstack()
    proportions = total_by_region.div(total_by_region.sum(axis=1), axis=0)
    
    plt.figure(figsize=(12, 6))
    
    # 繪製堆疊柱狀圖
    proportions.plot(kind='bar', stacked=True)
    
    plt.title('Energy Source Composition by Region', fontsize=14)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.legend(title='Energy Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 儲存圖表
    save_figure(plt, save_path, save_format, dpi)

def save_figure(plt, save_path=None, save_format='png', dpi=300):
    """
    儲存圖表的通用函數
    
    Args:
        plt: matplotlib.pyplot 物件
        save_path (str, optional): 圖表儲存路徑
        save_format (str): 圖片格式
        dpi (int): 圖片解析度
    """
    if save_path:
        # 確保儲存路徑存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 如果檔名沒有副檔名，加上指定的格式
        if not os.path.splitext(save_path)[1]:
            save_path = f"{save_path}.{save_format}"
            
        try:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format=save_format)
            print(f"圖表已儲存至：{save_path}")
        except Exception as e:
            print(f"儲存圖表時發生錯誤：{str(e)}")
    
    plt.show()
    plt.close()

def analyze_time_series_trends(df):
    """
    分析再生能源使用的時間趨勢
    
    Args:
        df (pandas.DataFrame): 原始資料框架
    """
    # 計算每年各地區的總使用量
    yearly_usage = df.groupby(['Region', 'Year'])['Monthly_Usage_kWh'].sum().reset_index()
    
    # 繪製時間趨勢圖
    plt.figure(figsize=(12, 6))
    
    for region in yearly_usage['Region'].unique():
        region_data = yearly_usage[yearly_usage['Region'] == region]
        plt.plot(region_data['Year'], region_data['Monthly_Usage_kWh'], 
                marker='o', label=region)
    
    # 新增：設定 x 軸刻度為整數年份
    plt.xticks(yearly_usage['Year'].unique())
    
    plt.title('Renewable Energy Usage Trends by Region (2020-2024)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Usage (kWh)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt

def analyze_energy_transition(df):
    """
    分析各地區能源結構的變化
    
    Args:
        df (pandas.DataFrame): 原始資料框架
    """
    # 計算每年各地區不同能源類型的佔比
    yearly_composition = df.groupby(['Region', 'Year', 'Energy_Source'])['Monthly_Usage_kWh'].sum().reset_index()
    total_by_region_year = yearly_composition.groupby(['Region', 'Year'])['Monthly_Usage_kWh'].sum().reset_index()
    
    # 合併並計算佔比
    yearly_composition = yearly_composition.merge(total_by_region_year, on=['Region', 'Year'], suffixes=('', '_total'))
    yearly_composition['Percentage'] = yearly_composition['Monthly_Usage_kWh'] / yearly_composition['Monthly_Usage_kWh_total'] * 100
    
    # 繪製轉型趨勢圖
    plt.figure(figsize=(15, 8))
    
    regions = yearly_composition['Region'].unique()
    years = yearly_composition['Year'].unique()
    n_regions = len(regions)
    n_cols = 3
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    # 用於存儲所有能源類型的列表，用於之後統一的圖例
    lines = []
    labels = []
    
    for idx, region in enumerate(regions):
        region_data = yearly_composition[yearly_composition['Region'] == region]
        
        for source in region_data['Energy_Source'].unique():
            source_data = region_data[region_data['Energy_Source'] == source]
            line = axes[idx].plot(source_data['Year'], source_data['Percentage'], 
                                marker='o', label=source)
            
            # 只在第一個子圖時收集圖例信息
            if idx == 0:
                lines.append(line[0])
                labels.append(source)
        
        axes[idx].set_xticks(years)
        axes[idx].set_title(f'{region}')
        axes[idx].set_xlabel('Year')
        axes[idx].set_ylabel('Percentage (%)')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        # 移除個別子圖的圖例
        axes[idx].get_legend().remove() if axes[idx].get_legend() else None
    
    # 隱藏多餘的子圖
    for idx in range(len(regions), len(axes)):
        axes[idx].set_visible(False)
    
    # 在整張圖的右上角添加統一的圖例
    fig.legend(lines, labels, 
              loc='upper right',
              bbox_to_anchor=(0.99, 0.99),
              title='Energy Source')
    
    plt.suptitle('Energy Source Transition by Region (2020-2024)', fontsize=16)
    plt.tight_layout()
    
    return plt

def calculate_diversity_index(df):
    """
    計算各地區的能源多樣性指標（Shannon 指數）
    
    Args:
        df (pandas.DataFrame): 原始資料框架
    Returns:
        pandas.Series: 各地區的 Shannon 多樣性指數
    """
    # 計算各地區各能源類型的總使用量
    energy_dist = df.groupby(['Region', 'Energy_Source'])['Monthly_Usage_kWh'].sum()
    
    # 計算比例
    total_by_region = energy_dist.groupby('Region').sum()
    proportions = energy_dist.unstack().div(total_by_region, axis=0)
    
    # 計算 Shannon 指數
    shannon_index = -(proportions * np.log(proportions)).sum(axis=1)
    
    return shannon_index

def analyze_growth_stability(df):
    """
    分析各地區能源使用成長的穩定性
    
    Args:
        df (pandas.DataFrame): 原始資料框架
    Returns:
        pandas.DataFrame: 包含成長率統計的資料框
    """
    # 計算年度總使用量
    yearly = df.groupby(['Region', 'Year'])['Monthly_Usage_kWh'].sum().reset_index()
    
    # 計算年度成長率
    growth_rates = yearly.pivot(index='Region', columns='Year', values='Monthly_Usage_kWh').pct_change(axis=1)
    
    # 計算統計指標
    growth_stats = pd.DataFrame({
        'Mean_Growth': growth_rates.mean(axis=1),
        'Std_Growth': growth_rates.std(axis=1),
        'Max_Growth': growth_rates.max(axis=1),
        'Min_Growth': growth_rates.min(axis=1)
    })
    
    return growth_stats

def main():
    """
    主程式流程
    """
    # 設定檔案路徑
    file_path = '../data/raw/Renewable_Energy_Usage_Sampled.csv'
    output_dir = '../output/figures/region_analysis'
    
    # 讀取資料
    df = load_data(file_path)
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 執行各項分析
    # 1. 時間趨勢分析
    time_series_plot = analyze_time_series_trends(df)
    time_series_plot.savefig(f'{output_dir}/time_series_trends.png', 
                            bbox_inches='tight', dpi=300)
    
    # 2. 能源轉型分析
    transition_plot = analyze_energy_transition(df)
    transition_plot.savefig(f'{output_dir}/energy_transition.png', 
                           bbox_inches='tight', dpi=300)
    
    # 3. 原有的地區分析
    region_year_source, regions, years, energy_sources, data = sort_region_data(df)
    plot_region_data(
        region_year_source, regions, years, energy_sources, data,
        f'{output_dir}/energy_composition'
    )
    
    # 計算並顯示統計摘要
    summary = calculate_region_summary(region_year_source)
    print("\nRegion Summary Statistics:")
    print(summary)
    
    # 新增分析
    diversity_index = calculate_diversity_index(df)
    growth_stats = analyze_growth_stability(df)
    
    # 輸出結果
    print("\nEnergy Diversity Index by Region:")
    print(diversity_index)
    
    print("\nGrowth Statistics by Region:")
    print(growth_stats)

if __name__ == "__main__":
    main()


