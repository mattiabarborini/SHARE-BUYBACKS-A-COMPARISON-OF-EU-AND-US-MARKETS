import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load panel data from Excel sheet into a DataFrame
file_path = '/Users/mattiabarborini/Downloads/updated_combined_results.xlsx'
panel_data = pd.read_excel(file_path)

# Remove extra spaces from column names
panel_data.columns = panel_data.columns.str.strip()

# Convert 'time period' to datetime format
panel_data['time period'] = pd.to_datetime(panel_data['time period'], format='%b-%Y', errors='coerce')

# Drop rows with missing or infinite values in specified columns
panel_data = panel_data.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
panel_data = panel_data.dropna(subset=['MKT CAP', 'D1', 'D2', 'D1D2', 'R^2'], how='any')

# Perform Difference-in-Differences regression using statsmodels
model = sm.OLS(panel_data['R^2'], sm.add_constant(panel_data[['D1', 'D2', 'D1D2', 'MKT CAP']])).fit()

# Create a figure and axes to control margins
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figsize as needed

# Plot the interaction effect (D1D2) on R^2 using scatterplot
sns.scatterplot(x='D1D2', y='R^2', hue='D1', data=panel_data, ax=ax)
plt.title('Interaction Effect (D1D2) on R^2', loc='left', pad=20)  # Adjust title position to the left
plt.xlabel('D1D2 Interaction')
plt.ylabel('R^2')
plt.tight_layout()  # Adjusts plot elements for better layout
plt.show()

# Save regression summary to an Excel file
summary_to_excel = model.summary().tables[1]
summary_df = pd.DataFrame(summary_to_excel.data)
summary_df.to_excel('DIF_in_DIF_panel_data.xlsx', index=False)

# Create a comparative graph for buyback vs. non-buyback samples
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figsize as needed
sns.boxplot(x='D1', y='R^2', data=panel_data, ax=ax)
plt.title('Comparison of Buyback (Treatment) and Non-Buyback (Control) Samples', loc='left', pad=20)  # Adjust title position to the left
plt.xlabel('Buyback Status')
plt.ylabel('R^2')
plt.tight_layout()  # Adjusts plot elements for better layout
plt.show()









# Histogram of R^2 by D1 (Geography)
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data=panel_data, x='R^2', hue='D1', multiple='stack', kde=True)
plt.title('Distribution of R^2 by Geography (D1)', loc='left', pad=10)  # Adjusting title position
plt.xlabel('R^2')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Histogram of R^2 by D2 (Treatment/Control)
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data=panel_data, x='R^2', hue='D2', multiple='stack', kde=True)
plt.title('Distribution of R^2 by Treatment/Control (D2)', loc='left', pad=10)  # Adjusting title position
plt.xlabel('R^2')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Boxplot of R^2 by D1 and D2
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='D1', y='R^2', hue='D2', data=panel_data)
plt.title('Boxplot of R^2 by Geography (D1) and Treatment/Control (D2)', loc='left', pad=10)  # Adjusting title position
plt.xlabel('Geography (D1)')
plt.ylabel('R^2')
plt.tight_layout()
plt.show()

# Violin Plot of R^2 by D1 and D2
fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x='D1', y='R^2', hue='D2', data=panel_data, split=True)
plt.title('Violin Plot of R^2 by Geography (D1) and Treatment/Control (D2)', loc='left', pad=10)  # Adjusting title position
plt.xlabel('Geography (D1)')
plt.ylabel('R^2')
plt.tight_layout()
plt.show()

# Facet Grid of Scatterplots for D1D2 vs R^2
g = sns.FacetGrid(panel_data, col='D1', row='D2', margin_titles=True)
g.map(sns.scatterplot, 'D1D2', 'R^2')
g.fig.suptitle('Scatterplots of D1D2 vs R^2 for Different Groups', x=0.5, y=1.02)  # Adjusting title position
plt.tight_layout()
plt.show()




import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load panel data from Excel sheet into a DataFrame
file_path = '/Users/mattiabarborini/Downloads/updated_combined_results.xlsx'
panel_data = pd.read_excel(file_path)

# Remove extra spaces from column names
panel_data.columns = panel_data.columns.str.strip()

# Convert 'time period' to datetime format
panel_data['time period'] = pd.to_datetime(panel_data['time period'], format='%b-%Y', errors='coerce')

# Drop rows with missing or infinite values in specified columns
panel_data = panel_data.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
panel_data = panel_data.dropna(subset=['MKT CAP', 'D1', 'D2', 'D1D2', 'R^2'], how='any')

# Perform Difference-in-Differences regression using statsmodels
model = sm.OLS(panel_data['R^2'], sm.add_constant(panel_data[['D1', 'D2', 'D1D2', 'MKT CAP']])).fit()

# Create a figure and axes to control margins
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figsize as needed

# Plot the interaction effect (D1D2) on R^2 using scatterplot
sns.scatterplot(x='D1D2', y='R^2', hue='D1', data=panel_data, ax=ax)
plt.title('Interaction Effect (D1D2) on R^2', loc='left', pad=20)  # Adjust title position to the left
plt.xlabel('D1D2 Interaction')
plt.ylabel('R^2')
plt.tight_layout()  # Adjusts plot elements for better layout
plt.show()

# Save regression summary to an Excel file
summary_to_excel = model.summary().tables[1]
summary_df = pd.DataFrame(summary_to_excel.data)
summary_df.to_excel('regression_summary.xlsx', index=False)

# Create a comparative graph for buyback vs. non-buyback samples
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figsize as needed
sns.boxplot(x='D1', y='R^2', data=panel_data, ax=ax)
plt.title('Comparison of Buyback (Treatment) and Non-Buyback (Control) Samples', loc='left', pad=20)  # Adjust title position to the left
plt.xlabel('Buyback Status')
plt.ylabel('R^2')
plt.tight_layout()  # Adjusts plot elements for better layout
plt.show()

# Facet Grid of Scatterplots for D1D2 vs R^2
g = sns.FacetGrid(panel_data, col='D1', row='D2', margin_titles=True)
g.map(sns.scatterplot, 'D1D2', 'R^2')
g.fig.suptitle('Scatterplots of D1D2 vs R^2 for Different Groups', x=0.5, y=1.05)  # Adjusting title position
g.fig.subplots_adjust(top=0.9)  # Adjusting top margin to fit the title within the figure
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load panel data from Excel sheet into a DataFrame
file_path = '/Users/mattiabarborini/Downloads/updated_combined_results.xlsx'
panel_data = pd.read_excel(file_path)

# Data cleaning and preprocessing (similar to previous code)
# ...

# Assuming 'D1' represents the buyback indicator (1 for buyback, 0 for non-buyback)
buyback = panel_data[panel_data['D1'] == 1]['R^2']
non_buyback = panel_data[panel_data['D1'] == 0]['R^2']

# Perform t-test for independent samples
t_stat, p_value = ttest_ind(buyback, non_buyback, equal_var=False)  # Assuming unequal variances

# Set up the bar plot
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate means
means = [buyback.mean(), non_buyback.mean()]

# Plot the bar graph with error bars for standard error
sns.barplot(x=['Buyback', 'Non-Buyback'], y=means, ax=ax)
plt.errorbar(x=['Buyback', 'Non-Buyback'], y=means, yerr=[buyback.sem(), non_buyback.sem()], fmt='none', capsize=10)

plt.title('Difference in R-squared: Buyback vs. Non-Buyback Stocks')
plt.ylabel('R-squared')
plt.xlabel('Group')
plt.ylim(0, panel_data['R^2'].max() + 0.05)  # Adjust ylim if needed

# Indicate statistical significance with asterisks
if p_value < 0.05:
    plt.text(0, max(means) + 0.02, '*', fontsize=12, ha='center', va='bottom')  # Annotate the bar with an asterisk

plt.show()


# In statistical graphs such as bar plots or boxplots, asterisks (*) are often used to denote statistical significance. When an asterisk is placed above a group or bar in a plot, it typically indicates that there is a statistically significant difference between that group and another (or multiple other) groups being compared.
#
# In your context, where you see an asterisk above the 'Buyback' group's bar and not above the 'Non-Buyback' group's bar in a bar plot or boxplot, it suggests that there is a statistically significant difference in the mean or median (depending on the plot type) of the R-squared values between the 'Buyback' and 'Non-Buyback' groups.
#
# It's essential to note that the number of asterisks can sometimes indicate the level of significance. For instance:
#
# One asterisk (*) might indicate a significance level of p < 0.05.
# Two asterisks (**) might indicate a significance level of p < 0.01.
# Three asterisks (***) might indicate a significance level of p < 0.001.
# However, the specific convention for denoting significance levels can vary between studies or presentations, so it's important to check the legend or accompanying information to understand the exact significance level indicated by the asterisk in your specific plot.