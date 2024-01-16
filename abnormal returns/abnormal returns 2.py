import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
file_path = '/Users/mattiabarborini/Downloads/ABNORMAL_RETURNS_top.xlsx'
df_main = pd.read_excel(file_path, sheet_name='Sheet1')

# Load the shorter columns data
df_shorter = pd.read_excel(file_path, sheet_name='Sheet2')

# Merge the shorter columns data with the main dataframe based on the common columns 'Stock' and 'TICKER'
merged_df = df_main.merge(df_shorter[['Stock', 'tstatalpha', 'tstatbeta']], left_on='TICKER', right_on='Stock', how='left')

# Update 'T stat alpha' and 'T stat beta' columns with values from the shorter dataframe
mask_alpha = merged_df['T stat alpha'].isnull()
mask_beta = merged_df['T stat beta'].isnull()

merged_df.loc[mask_alpha, 'T stat alpha'] = merged_df.loc[mask_alpha, 'tstatalpha'].replace('[^0-9.-]', '', regex=True).astype(float)
merged_df.loc[mask_beta, 'T stat beta'] = merged_df.loc[mask_beta, 'tstatbeta'].replace('[^0-9.-]', '', regex=True).astype(float)

merged_df['T stat alpha'].fillna(merged_df['T stat alpha'].mean(), inplace=True)
merged_df['T stat beta'].fillna(merged_df['T stat beta'].mean(), inplace=True)

merged_df.columns = merged_df.columns.str.strip()

merged_df['alpha'] = pd.to_numeric(merged_df['alpha'], errors='coerce')
merged_df['beta'] = pd.to_numeric(merged_df['beta'], errors='coerce')
merged_df['market returns'] = pd.to_numeric(merged_df['market returns'], errors='coerce')

merged_df.loc[merged_df['TICKER'] == 'MLM', 'alpha'] = 0.000260689536

alpha_values = {
    'EFX': -0.00108216435,
    'PCAR': -0.000812540270,
    'GRMN': -0.000303605032
}

for stock, alpha_value in alpha_values.items():
    merged_df.loc[merged_df['TICKER'] == stock, 'alpha'] = alpha_value

# Calculate 'expected returns' based on the conditions
alpha_condition = abs(merged_df['T stat alpha']) > 1.96
beta_condition = abs(merged_df['T stat beta']) > 1.96

expected_returns_mask = (alpha_condition | beta_condition)

merged_df.loc[expected_returns_mask, 'expected returns'] = (
    merged_df.loc[expected_returns_mask, 'alpha'].astype(float) +
    merged_df.loc[expected_returns_mask, 'beta'].astype(float) * merged_df.loc[expected_returns_mask, 'market returns'].astype(float)
)

merged_df['expected returns'].fillna(0, inplace=True)

merged_df['abnormal returns'] = merged_df['actual returns'] - merged_df['expected returns']

merged_df.to_excel('panel_data_abnormal_returns.xlsx', index=False)

mlm_data = merged_df[merged_df['TICKER'] == 'MLM']
relevant_columns = ['TICKER', 'alpha', 'beta', 'market returns', 'T stat alpha', 'T stat beta', 'expected returns', 'abnormal returns']
print(mlm_data[relevant_columns])

# Separate buyback and non-buyback stocks based on 'D1' column
buyback_stocks = merged_df[merged_df['D1'] == 1]
non_buyback_stocks = merged_df[merged_df['D1'] == 0]

# Box plot to compare abnormal returns for buyback and non-buyback stocks
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.boxplot([buyback_stocks['abnormal returns'], non_buyback_stocks['abnormal returns']], labels=['Buyback Stocks', 'Non-Buyback Stocks'])
plt.title('Abnormal Returns for Buyback vs. Non-Buyback Stocks')
plt.xlabel('Stock Groups')
plt.ylabel('Abnormal Returns')
plt.show()


# Calculate mean abnormal returns for buyback and non-buyback stocks
mean_abnormal_returns_buyback = buyback_stocks['abnormal returns'].mean()
mean_abnormal_returns_non_buyback = non_buyback_stocks['abnormal returns'].mean()

# Bar plot to compare mean abnormal returns for buyback and non-buyback stocks
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.bar(['Buyback Stocks', 'Non-Buyback Stocks'], [mean_abnormal_returns_buyback, mean_abnormal_returns_non_buyback])
plt.title('Mean Abnormal Returns for Buyback vs. Non-Buyback Stocks')
plt.xlabel('Stock Groups')
plt.ylabel('Mean Abnormal Returns')
plt.show()

import random

# Get a list of unique stock TICKERs
unique_tickers = merged_df['TICKER'].unique()

# Select 50 random tickers
random.seed(42)  # Set a seed for reproducibility
selected_tickers = random.sample(list(unique_tickers), 50)

# Filter selected tickers that are present in the data
existing_tickers = [ticker for ticker in selected_tickers if ticker in unique_tickers]

# Separate data for buyback and non-buyback periods for each stock
before_buyback = merged_df[merged_df['D1'] == 0]
after_buyback = merged_df[merged_df['D1'] == 1]

# Group by TICKER and calculate mean abnormal returns before and after buyback for selected stocks
mean_abnormal_returns_before = before_buyback[before_buyback['TICKER'].isin(existing_tickers)].groupby('TICKER')['abnormal returns'].mean()
mean_abnormal_returns_after = after_buyback[after_buyback['TICKER'].isin(existing_tickers)].groupby('TICKER')['abnormal returns'].mean()

# Plotting the mean abnormal returns for selected tickers
plt.figure(figsize=(10, 6))

# Plot mean abnormal returns before buyback
plt.plot(mean_abnormal_returns_before, marker='o', label='Mean Before Buyback')

# Plot mean abnormal returns after buyback
plt.plot(mean_abnormal_returns_after, marker='o', label='Mean After Buyback')

plt.title('Mean Abnormal Returns Before and After Buyback')
plt.xlabel('Stocks')
plt.ylabel('Mean Abnormal Returns')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

# Separate data for buyback and non-buyback periods for each stock
before_buyback = merged_df[merged_df['D1'] == 0]
after_buyback = merged_df[merged_df['D1'] == 1]

# Calculate mean abnormal returns before and after buyback for all available data
overall_mean_abnormal_returns_before = before_buyback.groupby('TICKER')['abnormal returns'].mean().mean()
overall_mean_abnormal_returns_after = after_buyback.groupby('TICKER')['abnormal returns'].mean().mean()

# Create a bar plot to compare the means
plt.figure(figsize=(8, 6))
plt.bar(['Before Buyback', 'After Buyback'], [overall_mean_abnormal_returns_before, overall_mean_abnormal_returns_after], color=['blue', 'orange'])
plt.title('Comparison of Mean Abnormal Returns Before and After Buybacks (All Data)')
plt.xlabel('Period')
plt.ylabel('Overall Mean Abnormal Returns')
plt.show()










# Separate data for buyback and non-buyback periods for each stock
before_buyback = merged_df[merged_df['D1'] == 0]
after_buyback = merged_df[merged_df['D1'] == 1]

# Calculate mean of mean abnormal returns before and after buyback for all available data
mean_of_mean_before_buyback = before_buyback.groupby('TICKER')['abnormal returns'].mean().mean()
mean_of_mean_after_buyback = after_buyback.groupby('TICKER')['abnormal returns'].mean().mean()

# Create a bar plot to compare the means
plt.figure(figsize=(8, 6))
plt.bar(['Before Buyback', 'After Buyback'], [mean_of_mean_before_buyback, mean_of_mean_after_buyback], color=['blue', 'orange'])
plt.title('Comparison of Mean of Mean Abnormal Returns Before and After Buybacks (All Data)')
plt.xlabel('Period')
plt.ylabel('Mean of Mean Abnormal Returns')
plt.show()





# Separate buyback and non-buyback stocks based on 'D1' column
buyback_stocks = merged_df[merged_df['D1'] == 1]
non_buyback_stocks = merged_df[merged_df['D1'] == 0]

# Calculate mean abnormal returns for buyback and non-buyback stocks
mean_abnormal_returns_buyback = buyback_stocks['abnormal returns'].mean()
mean_abnormal_returns_non_buyback = non_buyback_stocks['abnormal returns'].mean()

# Define colors for buyback and non-buyback bars
colors = ['orange', 'blue']

# Bar plot to compare mean abnormal returns for buyback and non-buyback stocks
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.bar(['D1 = 1 (Buyback Stocks)', 'D1 = 0 (Non-Buyback Stocks)'],
        [mean_abnormal_returns_buyback, mean_abnormal_returns_non_buyback],
        color=colors)
plt.title('Mean Abnormal Returns for Buyback vs. Non-Buyback Stocks')
plt.xlabel('D1 Values')
plt.ylabel('Mean Abnormal Returns')
plt.show()










