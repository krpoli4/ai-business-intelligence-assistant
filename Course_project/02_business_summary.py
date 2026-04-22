import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Course_project\Dataset\sales_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Extract time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Business metrics
sales_by_month = df.groupby("Month")["Sales"].sum()
sales_by_product = df.groupby("Product")["Sales"].sum()
sales_by_region = df.groupby("Region")["Sales"].sum()
sales_by_gender = df.groupby("Customer_Gender")["Sales"].mean()
sales_by_age = df.groupby("Customer_Age")["Sales"].mean()

median_sales = df["Sales"].median()
std_sales = df["Sales"].std()
avg_satisfaction = df["Customer_Satisfaction"].mean()

# Create summary text for LLM
summary_text = f"""
Business Intelligence Summary

1. Sales by Month:
{sales_by_month.to_string()}

2. Sales by Product:
{sales_by_product.to_string()}

3. Sales by Region:
{sales_by_region.to_string()}

4. Average Sales by Gender:
{sales_by_gender.to_string()}

5. Average Sales by Age:
{sales_by_age.head(10).to_string()}

6. Statistical Measures:
Median Sales: {median_sales}
Standard Deviation of Sales: {std_sales}
Average Customer Satisfaction: {avg_satisfaction}
"""

print(summary_text)