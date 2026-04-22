import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Course_project\Dataset\sales_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Extract time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Preview data
print(df.head())

print(df.info())

print(df.describe())

# Business Metrics
sales_by_month = df.groupby("Month")["Sales"].sum()
sales_by_product = df.groupby("Product")["Sales"].sum()
sales_by_region = df.groupby("Region")["Sales"].sum()

sales_by_age = df.groupby("Customer_Age")["Sales"].mean()

median_sales = df["Sales"].median()
std_sales = df["Sales"].std()

print("\nSales by Month:")
print(sales_by_month)

print("\nSales by Product:")
print(sales_by_product)

print("\nSales by Region:")
print(sales_by_region)

print("\nMedian Sales:", median_sales)
print("Standard Deviation:", std_sales)

# Visualizations

sales_by_month.plot(kind="line", marker="o")
plt.title("Sales Trend Over Months")
plt.show()

sales_by_product.plot(kind="bar")
plt.title("Sales by Product")
plt.show()

sales_by_region.plot(kind="bar")
plt.title("Sales by Region")
plt.show()

sales_by_age.plot(kind="line")
plt.title("Average Sales by Customer Age")
plt.show()