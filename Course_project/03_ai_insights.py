from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load API key from .env
load_dotenv()

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

# Create summary text
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

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a business intelligence assistant. "
     "Analyze the provided business summary and return: "
     "1. Key trends "
     "2. Top-performing product "
     "3. Best-performing region "
     "4. Customer insights "
     "5. Three business recommendations. "
     "Keep the response clear and professional."),
    ("human", "{data}")
])

# Model
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# Invoke
formatted_prompt = prompt.invoke({"data": summary_text})
response = llm.invoke(formatted_prompt)

print("\n===== AI BUSINESS INSIGHTS =====\n")
print(response.content)