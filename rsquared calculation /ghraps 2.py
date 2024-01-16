import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load regression results data from Excel files
sp_results = pd.read_excel('Regression_Results_S&P_Stats_Numbers.xlsx')
eu_results = pd.read_excel('Regression_Results_Eurostoxx_Stats_Numbers.xlsx')

# Plotting R-squared comparison between S&P and Eurostoxx
plt.figure(figsize=(10, 6))
sns.histplot(sp_results['R-squared'], color='blue', label='S&P', kde=True)
sns.histplot(eu_results['R-squared'], color='red', label='Eurostoxx', kde=True)
plt.xlabel('R-squared')
plt.ylabel('Frequency')
plt.title('Comparison of R-squared between S&P and Eurostoxx')
plt.legend()
plt.show()

# Plotting Beta comparison between S&P and Eurostoxx
plt.figure(figsize=(10, 6))
sns.histplot(sp_results['Beta'], color='blue', label='S&P', kde=True)
sns.histplot(eu_results['Beta'], color='red', label='Eurostoxx', kde=True)
plt.xlabel('Beta')
plt.ylabel('Frequency')
plt.title('Comparison of Beta between S&P and Eurostoxx')
plt.legend()
plt.show()




import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load panel data from Excel sheet into a DataFrame
file_path = '/Users/mattiabarborini/Downloads/updated_combined_results.xlsx'
panel_data = pd.read_excel(file_path)

# Data cleaning and preprocessing (similar to the previous code)
# ... (Code for data cleaning, preprocessing, and regression analysis)

# Create a pair plot to visualize relationships between different variables
sns.pairplot(panel_data[['D1', 'D2', 'D1D2', 'MKT CAP', 'R^2']])
plt.suptitle('Pair Plot of Variables', y=1.02)
plt.show()

# Create a heatmap to visualize correlation among variables
correlation_matrix = panel_data[['D1', 'D2', 'D1D2', 'MKT CAP', 'R^2']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Create violin plots to compare distribution of 'R^2' for different groups
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.violinplot(x='D1', y='R^2', data=panel_data, ax=axes[0])
axes[0].set_title('Buyback (D1) vs. R^2')

sns.violinplot(x='D2', y='R^2', data=panel_data, ax=axes[1])
axes[1].set_title('Treatment (D2) vs. R^2')
plt.tight_layout()
plt.show()

# Create a jointplot to visualize the interaction effect (D1D2) on 'R^2'
sns.jointplot(x='D1D2', y='R^2', data=panel_data, kind='reg')
plt.subplots_adjust(top=0.95)
plt.suptitle('Interaction Effect (D1D2) on R^2')
plt.show()


import pandas as pd
import plotly.express as px

# Load panel data from Excel sheet into a DataFrame
file_path = '/Users/mattiabarborini/Downloads/updated_combined_results.xlsx'
panel_data = pd.read_excel(file_path)

# Data cleaning and preprocessing
panel_data.columns = panel_data.columns.str.strip()
panel_data['time period'] = pd.to_datetime(panel_data['time period'], format='%b-%Y', errors='coerce')
panel_data['MKT CAP'] = pd.to_numeric(panel_data['MKT CAP'], errors='coerce')
panel_data.dropna(subset=['MKT CAP', 'D1', 'D2', 'D1D2', 'R^2'], inplace=True)

# Assuming 'D1' represents the buyback indicator (1 for buyback, 0 for non-buyback)
panel_data['Buyback_Status'] = panel_data['D1'].map({1: 'Buyback', 0: 'Non-Buyback'})

# Create an interactive scatter plot using Plotly Express
fig = px.scatter(panel_data, x='D1D2', y='R^2', color='Buyback_Status', size='MKT CAP',
                 hover_data=['D1', 'D2'], title='Comparison of R-squared between Buyback and Non-Buyback Stocks',
                 labels={'D1D2': 'Interaction Effect', 'R^2': 'R-squared'})

# Update layout for better visibility
fig.update_layout(title={'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'},
                  xaxis_title='Interaction Effect (D1D2)',
                  yaxis_title='R-squared', legend_title='Buyback Status')

fig.show()



import pandas as pd
import plotly.express as px

# Load panel data from Excel sheet into a DataFrame
file_path = '/Users/mattiabarborini/Downloads/updated_combined_results.xlsx'
panel_data = pd.read_excel(file_path)

# Data cleaning and preprocessing
panel_data.columns = panel_data.columns.str.strip()
panel_data['time period'] = pd.to_datetime(panel_data['time period'], format='%b-%Y', errors='coerce')
panel_data['MKT CAP'] = pd.to_numeric(panel_data['MKT CAP'], errors='coerce')
panel_data.dropna(subset=['MKT CAP', 'D1', 'D2', 'D1D2', 'R^2'], inplace=True)

# Assuming 'D1' represents the buyback indicator (1 for buyback, 0 for non-buyback)
panel_data['Buyback_Status'] = panel_data['D1'].map({1: 'Buyback', 0: 'Non-Buyback'})

# Create an interactive boxplot using Plotly Express
fig = px.box(panel_data, x='Buyback_Status', y='R^2', color='Buyback_Status',
             labels={'Buyback_Status': 'Buyback Status', 'R^2': 'R-squared'},
             title='Comparison of R-squared between Buyback and Non-Buyback Stocks')

# Update layout for better visibility
fig.update_layout(title={'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'},
                  xaxis_title='Buyback Status', yaxis_title='R-squared')

fig.show()




