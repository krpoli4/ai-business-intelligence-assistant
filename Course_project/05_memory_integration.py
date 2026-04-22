from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load API key
load_dotenv()

# Load dataset
df = pd.read_csv(r"C:\Course_project\Dataset\sales_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Precompute metrics
sales_by_month = df.groupby("Month")["Sales"].sum()
sales_by_product = df.groupby("Product")["Sales"].sum()
sales_by_region = df.groupby("Region")["Sales"].sum()
sales_by_gender = df.groupby("Customer_Gender")["Sales"].mean()
sales_by_age = df.groupby("Customer_Age")["Sales"].mean()

median_sales = df["Sales"].median()
std_sales = df["Sales"].std()
avg_satisfaction = df["Customer_Satisfaction"].mean()

# Simple retriever
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

# Store conversation history in memory
conversation_history = []

# Prompt with memory
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Business Intelligence Assistant. "
        "Use the retrieved business data and prior conversation history to answer clearly. "
        "Answer only from the provided context and history. "
        "If the answer is not supported by the provided context, say the available data does not contain enough information."
    ),
    (
        "human",
        "Conversation History:\n{history}\n\n"
        "Current Question:\n{question}\n\n"
        "Retrieved Context:\n{context}"
    )
])

# Model
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

print("Business Intelligence Assistant with Memory is ready.")
print("Type 'exit' to quit.\n")

while True:
    user_question = input("Ask a question: ")

    if user_question.lower() == "exit":
        print("Exiting assistant.")
        break

    # Retrieve context
    context = retrieve_context(user_question)

    # Build history text
    history_text = "\n".join(conversation_history[-6:])  # last 6 exchanges

    # Invoke model
    formatted_prompt = prompt.invoke({
        "history": history_text if history_text else "No prior conversation.",
        "question": user_question,
        "context": context
    })

    response = llm.invoke(formatted_prompt)
    answer = response.content

    print("\nAnswer:")
    print(answer)
    print("\n" + "-" * 50 + "\n")

    # Save to memory
    conversation_history.append(f"User: {user_question}")
    conversation_history.append(f"Assistant: {answer}")