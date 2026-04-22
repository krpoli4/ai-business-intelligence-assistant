from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load API key
load_dotenv()

# Load dataset
df = pd.read_csv(r"C:\Course_project\Dataset\sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Metrics
sales_by_month = df.groupby("Month")["Sales"].sum()
sales_by_product = df.groupby("Product")["Sales"].sum()
sales_by_region = df.groupby("Region")["Sales"].sum()
sales_by_gender = df.groupby("Customer_Gender")["Sales"].mean()
sales_by_age = df.groupby("Customer_Age")["Sales"].mean()

median_sales = df["Sales"].median()
std_sales = df["Sales"].std()
avg_satisfaction = df["Customer_Satisfaction"].mean()

# Retriever
def retrieve_context(question: str) -> str:
    q = question.lower()

    if "month" in q or "sales trend" in q or "time period" in q:
        return f"Sales by Month:\n{sales_by_month.to_string()}"

    elif "product" in q or "top product" in q or "best product" in q:
        return f"Sales by Product:\n{sales_by_product.to_string()}"

    elif "region" in q or "best region" in q or "regional" in q:
        return f"Sales by Region:\n{sales_by_region.to_string()}"

    elif "gender" in q:
        return f"Average Sales by Gender:\n{sales_by_gender.to_string()}"

    elif "age" in q or "customer segment" in q or "demographic" in q:
        return f"Average Sales by Age:\n{sales_by_age.to_string()}"

    elif "satisfaction" in q:
        return f"Average Customer Satisfaction: {avg_satisfaction}"

    elif "median" in q or "standard deviation" in q or "stats" in q or "statistics" in q:
        return (
            f"Median Sales: {median_sales}\n"
            f"Standard Deviation of Sales: {std_sales}"
        )

    else:
        return f"""
General Business Summary:

Sales by Month:
{sales_by_month.to_string()}

Sales by Product:
{sales_by_product.to_string()}

Sales by Region:
{sales_by_region.to_string()}

Average Sales by Gender:
{sales_by_gender.to_string()}

Average Sales by Age:
{sales_by_age.head(15).to_string()}

Median Sales: {median_sales}
Standard Deviation of Sales: {std_sales}
Average Customer Satisfaction: {avg_satisfaction}
"""

# QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Business Intelligence Assistant. "
        "Answer only from the provided context. "
        "If the answer is not in the context, say the available data does not contain enough information."
    ),
    (
        "human",
        "Question: {question}\n\nRetrieved Context:\n{context}"
    )
])

# Evaluation prompt
eval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an evaluator for a business intelligence QA system. "
        "Compare the expected answer and the model answer. "
        "Return only one word: CORRECT, PARTIALLY_CORRECT, or INCORRECT."
    ),
    (
        "human",
        "Question: {question}\n\nExpected Answer:\n{expected}\n\nModel Answer:\n{actual}"
    )
])

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# Test set
test_cases = [
    {
        "question": "Which product performed best?",
        "expected": "Widget A performed best with total sales of 375235."
    },
    {
        "question": "Which region had the highest sales?",
        "expected": "West region had the highest sales with 361383."
    },
    {
        "question": "What is the average customer satisfaction?",
        "expected": "Average customer satisfaction is approximately 3.03."
    },
    {
        "question": "What is the median sales value?",
        "expected": "Median sales value is 552.5."
    },
    {
        "question": "Which month had the highest sales?",
        "expected": "Month 8 had the highest sales with 124264."
    }
]

results = []

for case in test_cases:
    question = case["question"]
    expected = case["expected"]

    context = retrieve_context(question)

    qa_input = qa_prompt.invoke({
        "question": question,
        "context": context
    })
    qa_response = llm.invoke(qa_input)
    actual = qa_response.content.strip()

    eval_input = eval_prompt.invoke({
        "question": question,
        "expected": expected,
        "actual": actual
    })
    eval_response = llm.invoke(eval_input)
    verdict = eval_response.content.strip()

    results.append({
        "Question": question,
        "Expected Answer": expected,
        "Model Answer": actual,
        "Evaluation": verdict
    })

# Save results
results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv(r"C:\Course_project\evaluation_results.csv", index=False)
print("\nEvaluation results saved to C:\\Course_project\\evaluation_results.csv")