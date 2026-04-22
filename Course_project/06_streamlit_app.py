from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")

st.title("InsightForge: AI-Powered Business Intelligence Assistant")
st.write("Ask business questions, explore sales trends, and view key insights.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Course_project\Dataset\sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    return df

df = load_data()

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

# Memory
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Prompt
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

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# Sidebar charts
st.sidebar.header("Business Visualizations")

chart_option = st.sidebar.selectbox(
    "Select Chart",
    [
        "Sales by Month",
        "Sales by Product",
        "Sales by Region",
        "Average Sales by Age"
    ]
)

fig, ax = plt.subplots()

if chart_option == "Sales by Month":
    sales_by_month.plot(kind="line", marker="o", ax=ax)
    ax.set_title("Sales Trend Over Months")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")

elif chart_option == "Sales by Product":
    sales_by_product.plot(kind="bar", ax=ax)
    ax.set_title("Sales by Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Sales")

elif chart_option == "Sales by Region":
    sales_by_region.plot(kind="bar", ax=ax)
    ax.set_title("Sales by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Sales")

elif chart_option == "Average Sales by Age":
    sales_by_age.plot(kind="line", ax=ax)
    ax.set_title("Average Sales by Customer Age")
    ax.set_xlabel("Customer Age")
    ax.set_ylabel("Average Sales")

st.sidebar.pyplot(fig)

# Main question box
st.subheader("Ask a Business Question")
user_question = st.text_input("Enter your question:")

if st.button("Generate Answer") and user_question:
    context = retrieve_context(user_question)
    history_text = "\n".join(st.session_state.conversation_history[-6:])

    formatted_prompt = prompt.invoke({
        "history": history_text if history_text else "No prior conversation.",
        "question": user_question,
        "context": context
    })

    response = llm.invoke(formatted_prompt)
    answer = response.content

    st.session_state.conversation_history.append(f"User: {user_question}")
    st.session_state.conversation_history.append(f"Assistant: {answer}")

    st.subheader("AI Answer")
    st.write(answer)

    with st.expander("Retrieved Context Used"):
        st.text(context)

# Show conversation history
if st.session_state.conversation_history:
    st.subheader("Conversation History")
    for item in st.session_state.conversation_history:
        st.write(item)