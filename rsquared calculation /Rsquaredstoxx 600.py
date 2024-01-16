import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data from Excel file
file_path = '/Users/mattiabarborini/Downloads/RsquaredEurostoxx.xlsx'
sp500_data = pd.read_excel(file_path)

# Replace zero values in 'STOCK RETURNS 600' column with NaN
sp500_data['STOCK RETURNS 600'] = sp500_data['STOCK RETURNS 600'].replace(0, np.nan)

results_list = []  # Empty list to store results

# Loop through each stock in the Eurostoxx 600
for stock in sp500_data['TICKER 600'].unique():
    stock_data = sp500_data[sp500_data['TICKER 600'] == stock]

    # Filter out rows with NaN values in 'STOCK RETURNS 600' column
    stock_data = stock_data.dropna(subset=['STOCK RETURNS 600'])

    if len(stock_data) > 1:  # Ensure enough data points for regression
        # Perform regression analysis
        y = stock_data['STOCK RETURNS 600']
        X = sm.add_constant(stock_data['MARKET INDEX RETURNS 600'])
        model = sm.OLS(y, X).fit()

        # Get R-squared and beta (checking number of parameters in the model)
        r_squared = model.rsquared
        num_params = len(model.params)

        if num_params > 1:
            beta = model.params.iloc[1]  # The beta coefficient
        else:
            beta = None  # Set beta to None if there's no second parameter

        # Append results to the list
        results_list.append({'Stock': stock, 'R-squared': r_squared, 'Beta': beta})

# Convert the list of dictionaries into a DataFrame
results_df = pd.DataFrame(results_list)

# Save the DataFrame to an Excel file
results_df.to_excel('Regression_Results_Eurostoxx.xlsx', index=False)
