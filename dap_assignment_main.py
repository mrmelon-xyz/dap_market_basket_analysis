"""
Data Source: https://github.com/microsoft/powerbi-desktop-samples/blob/5cd33b34b544ad626a32f9b60c758b0e9ef25385/AdventureWorks%20Sales%20Sample/AdventureWorks%20Sales.xlsx
mlxtend:
        https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
        https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/

https://www.geeksforgeeks.org/implementing-apriori-algorithm-in-python/?ref=header_search

"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import time
import plotly.express as px

# Loading the Data
data = pd.read_excel('AdventureWorks_Sales.xlsx', sheet_name='Sales_data')
data.head()

# Exploring the columns of the data
print(data.columns)

# Exploring the different regions of transactions
print(data.OrderDateYear.unique())

# Stripping extra spaces in the Product Name
data['ProductName'] = data['ProductName'].str.strip()
print(data['ProductName'])

#  Initial Visualizations
data['Order Line Count'] = 1
data_table = data.groupby("ProductName").sum().sort_values("Order Line Count", ascending=False).reset_index()

data_table['All'] = 'Top 50 Products'
fig = px.treemap(data_table.head(50), path=['All', 'ProductName'], values='Order Line Count',
                  color=data_table["Order Line Count"].head(50), hover_data=['ProductName'],
                  color_continuous_scale='Blues',
                )
# ploting the treemap
### fig.show()

# Filter data for year 2020 and 2019
filter_values = [2020, 2019, 2018, 2017]

# basket = (data[(data['OrderDateYear'] == 2020)]
basket = (data[data['OrderDateYear'].isin(filter_values)]
             .groupby(['Sales Order', 'ProductName'])['Order Line Count']
             .sum().unstack().reset_index().fillna(0)
             .set_index('Sales Order'))


print(basket.head())

# Defining the hot encoding function to make the data suitable
# for the concerned libraries
def hot_encode(x):
	if(x<= 0):
		return False
	if(x>= 1):
		return True

# Encoding the datasets
basket_encoded = basket.map(hot_encode)
basket = basket_encoded
print(basket.shape[0])


### apriori
apriori_start_time = time.time()

frq_items_apriori = apriori(basket, min_support = 0.02, use_colnames = True)
print(frq_items_apriori.sort_values('support', ascending=False))
frq_items_apriori.sort_values('support', ascending=False).to_csv('dap_results_frq_items_apriori.csv')

rules = association_rules(frq_items_apriori, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head())
rules.to_csv('dap_results_rules_apriori.csv')

apriori_end_time = time.time()
apriori_elapsed_time = apriori_end_time - apriori_start_time

### fpgrowth
fpgrowth_start_time = time.time()

frq_items_fpgrowth = fpgrowth(basket, min_support = 0.02, use_colnames = True)
print(frq_items_fpgrowth.sort_values('support', ascending=False))
frq_items_fpgrowth.sort_values('support', ascending=False).to_csv('dap_results_frq_items_fpgrowth.csv')

rules = association_rules(frq_items_fpgrowth, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head())
rules.to_csv('dap_results_rules_fpgrowth.csv')

fpgrowth_end_time = time.time()
fpgrowth_elapsed_time = fpgrowth_end_time - fpgrowth_start_time

print(f'Apriori duration: {apriori_elapsed_time:.2f} seconds - FP Growth duration: {fpgrowth_elapsed_time:.2f} seconds')


