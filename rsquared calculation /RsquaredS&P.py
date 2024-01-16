import pandas as pd
import statsmodels.api as sm

# Load data from CSV files or your data source
file_path = '/Users/mattiabarborini/Downloads/RsquaredS&P.xlsx'
sp500_data = pd.read_excel(file_path)

results_list = []  # Empty list to store results

# Loop through each stock in the S&P 500
for stock in sp500_data['TICKER S&P'].unique():
    stock_data = sp500_data[sp500_data['TICKER S&P'] == stock]
    
    # Perform regression analysis
    y = stock_data['RETURNS S&P']
    X = sm.add_constant(stock_data['MARKET INDEX RETURNS S&P'])
    model = sm.OLS(y, X).fit()

    # Get R-squared and beta
    r_squared = model.rsquared
    beta = model.params.iloc[1]  # The beta coefficient

    # Append results to the list
    results_list.append({'Stock': stock, 'R-squared': r_squared, 'Beta': beta})

# Convert the list of dictionaries into a DataFrame
results_df = pd.DataFrame(results_list)

# Save the DataFrame to an Excel file
results_df.to_excel('Regression_Results_S&P.xlsx', index=False)
