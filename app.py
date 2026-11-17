"""
Chat-with-your-data BI agent  (Ollama + SQLite + Streamlit + Plotly)

No database account or server needed: SQLite is just a file (sales.db)
that this script creates automatically from sales_data.csv.
If sales_data.csv does not exist, sample data is generated for you.

Run:   streamlit run app.py
"""

import csv
import json
import os
import random
import re
import sqlite3
from datetime import date, timedelta

import ollama
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------- config ---------------------------------
MODEL = "llama3.2"          # any Ollama model that supports tool calling
CSV_PATH = "sales_data.csv"
DB_PATH = "sales.db"
TABLE = "sales"
MAX_STEPS = 6
MAX_ROWS_TO_LLM = 20

SYSTEM_PROMPT = f"""You are a BI analyst agent working on a SQLite database
with a table named {TABLE}.
Workflow:
1. The database schema is given below. Use ONLY those table and column names.
   Never create tables or invent columns.
2. Write ONE read-only SQLite SELECT query and call run_sql.
3. If run_sql returns an ERROR, read it, fix the SQL, and try again.
4. When the result suits a visual (comparison, trend, ranking), call make_chart.
5. Finish with a short 1-3 sentence plain-English answer based ONLY on the
   returned data. Never invent numbers.
Always use the tool-calling mechanism to call tools; never write tool JSON in your reply.
Dates are stored as text 'YYYY-MM-DD'; use strftime('%Y-%m', order_date) for months."""


# ----------------------- sample data + database ------------------------
def generate_sample_csv(path):
    random.seed(42)
    products = {
        "Technology": [("Laptop", 900), ("Smartphone", 650), ("Monitor", 220),
                       ("Keyboard", 45), ("Headphones", 80), ("Webcam", 60)],
        "Furniture": [("Office Chair", 180), ("Standing Desk", 420),
                      ("Bookcase", 150), ("Filing Cabinet", 120)],
        "Office Supplies": [("Notebook", 4), ("Printer Paper", 25), ("Pen Set", 12),
                            ("Stapler", 15), ("Binder", 8)],
    }
    regions = {"North": ["Delhi", "Chandigarh", "Lucknow"],
               "South": ["Bengaluru", "Chennai", "Hyderabad"],
               "East": ["Kolkata", "Bhubaneswar", "Guwahati"],
               "West": ["Mumbai", "Pune", "Nashik"]}
    segments = ["Consumer", "Corporate", "Home Office"]
    ships = ["Standard", "Express", "Same Day"]
    first = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Anjali", "Karan", "Meera",
             "Rahul", "Divya", "Arjun", "Neha", "Amit", "Pooja", "Sanjay", "Kavya"]
    last = ["Sharma", "Patil", "Reddy", "Nair", "Singh", "Das", "Joshi", "Mehta", "Iyer", "Gupta"]
    customers = [(f"C{1000+i}", f"{random.choice(first)} {random.choice(last)}",
                  random.choice(segments)) for i in range(60)]
    margins = {"Technology": 0.18, "Furniture": 0.10, "Office Supplies": 0.30}

    rows = []
    for n in range(1, 1201):
        d = date(2024, 1, 1) + timedelta(days=random.randint(0, 729))
        cid, cname, seg = random.choice(customers)
        region = random.choice(list(regions))
        city = random.choice(regions[region])
        cat = random.choices(list(products), weights=[3, 2, 5])[0]
        pname, base = random.choice(products[cat])
        qty = random.randint(1, 8) if cat == "Office Supplies" else random.randint(1, 3)
        disc = random.choice([0, 0, 0, 0.05, 0.1, 0.2])
        price = round(base * random.uniform(0.9, 1.1), 2)
        sales = round(price * qty * (1 - disc), 2)
        profit = round(sales * (margins[cat] + random.uniform(-0.12, 0.08))
                       - (sales * 0.25 if disc == 0.2 else 0), 2)
        rows.append([f"ORD-{n:05d}", d.isoformat(), cid, cname, seg, region, city, cat,
                     pname, price, qty, disc, sales, profit, random.choice(ships)])
    rows.sort(key=lambda r: r[1])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "order_date", "customer_id", "customer_name", "segment",
                    "region", "city", "category", "product", "unit_price", "quantity",
                    "discount", "sales", "profit", "ship_mode"])
        w.writerows(rows)


def setup_database():
    """Create sales.db from the CSV (generating the CSV first if missing)."""
    if os.path.exists(DB_PATH):
        return
    if not os.path.exists(CSV_PATH):
        generate_sample_csv(CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"]).dt.strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql(TABLE, con, if_exists="replace", index=False)


def connect_readonly():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


# ----------------------------- tools ----------------------------------
def get_schema(ctx):
    with connect_readonly() as con:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        sample = pd.read_sql_query(f"SELECT * FROM {TABLE} LIMIT 3", con).to_csv(index=False)
    return "\n\n".join(r[0] for r in rows) + f"\n\nSample rows:\n{sample}"


def run_sql(ctx, query):
    q = query.strip().rstrip(";")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE) or FORBIDDEN.search(q):
        return "ERROR: only read-only SELECT queries are allowed."
    try:
        with connect_readonly() as con:
            df = pd.read_sql_query(q, con)
    except Exception as e:
        return f"ERROR: {e}"
    ctx["df"], ctx["sql"] = df, q
    return f"{len(df)} rows. Columns: {list(df.columns)}\n{df.head(MAX_ROWS_TO_LLM).to_csv(index=False)}"


def make_chart(ctx, kind, x, y, title=""):
    df = ctx.get("df")
    if df is None:
        return "ERROR: run a SQL query first."
    if x not in df.columns or y not in df.columns:
        return f"ERROR: x and y must be among columns {list(df.columns)}"
    if kind == "bar":
        fig = px.bar(df, x=x, y=y, title=title)
    elif kind == "line":
        fig = px.line(df, x=x, y=y, title=title, markers=True)
    elif kind == "pie":
        fig = px.pie(df, names=x, values=y, title=title)
    elif kind == "scatter":
        fig = px.scatter(df, x=x, y=y, title=title)
    else:
        return "ERROR: kind must be bar, line, pie or scatter."
    ctx["fig"] = fig
    return "Chart created and shown to the user."


TOOLS = [
    {"type": "function", "function": {
        "name": "get_schema",
        "description": "Get the table definition and 3 sample rows.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run a read-only SQLite SELECT query. Returns rows or an ERROR.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "make_chart",
        "description": "Chart the most recent run_sql result.",
        "parameters": {"type": "object",
                       "properties": {
                           "kind": {"type": "string", "enum": ["bar", "line", "pie", "scatter"]},
                           "x": {"type": "string"},
                           "y": {"type": "string"},
                           "title": {"type": "string"}},
                       "required": ["kind", "x", "y"]}}},
]
REGISTRY = {"get_schema": get_schema, "run_sql": run_sql, "make_chart": make_chart}


# ----------------------------- agent loop -----------------------------
def parse_text_tool_call(content):
    """Fallback for small models that write the tool call as plain text,
    e.g. {"name":"run_sql","parameters":{"query":"SELECT ..."}}.
    Tolerates slightly broken JSON. Returns (name, args) or None."""
    if not content or '"name"' not in content:
        return None
    m = re.search(r'"name"\s*:\s*"(\w+)"', content)
    if not m or m.group(1) not in REGISTRY:
        return None
    name = m.group(1)
    try:  # clean JSON first
        obj = json.loads(content.strip())
        args = obj.get("parameters") or obj.get("arguments") or {}
        if isinstance(args, dict):
            return name, args
    except Exception:
        pass
    # broken JSON: pull out every "key":"value" pair
    pairs = re.findall(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
    return name, {k: v for k, v in pairs if k != "name"}


def run_agent(messages, ctx, on_step=None):
    for _ in range(MAX_STEPS):
        resp = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.message
        messages.append(msg)

        calls = [(c.function.name, dict(c.function.arguments)) for c in (msg.tool_calls or [])]
        if not calls:
            fallback = parse_text_tool_call(msg.content)
            if fallback:
                calls = [fallback]
            else:
                return msg.content or "(no answer)"

        for name, args in calls:
            fn = REGISTRY.get(name)
            try:
                result = fn(ctx, **args) if fn else f"ERROR: unknown tool {name}"
            except TypeError as e:
                result = f"ERROR: bad arguments for {name}: {e}"
            if on_step:
                on_step(name, args, result)
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})
    return "I couldn't finish within the step limit. Try rephrasing the question."


def build_system_prompt():
    """Put the schema straight into the prompt so small models don't guess."""
    return SYSTEM_PROMPT + "\n\nDATABASE SCHEMA:\n" + get_schema({})


# ----------------------------- UI -------------------------------------
def main():
    st.set_page_config(page_title="Chat with your data", page_icon="📊", layout="wide")
    st.title("📊 Chat with your data")
    st.caption(f"Ollama model: `{MODEL}` · Data: {CSV_PATH} (table `{TABLE}`)")
    setup_database()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": build_system_prompt()}]
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.subheader("Try asking")
        for ex in ["Which region has the highest profit?",
                   "Show monthly sales trend for 2025",
                   "Top 5 customers by total sales",
                   "Average profit by category and discount",
                   "Sales by ship mode as a pie chart"]:
            if st.button(ex, use_container_width=True):
                st.session_state.pending = ex
        if st.button("Clear chat"):
            st.session_state.messages = [{"role": "system", "content": build_system_prompt()}]
            st.session_state.history = []
            st.rerun()

    def render(turn):
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("sql"):
                with st.expander("SQL used"):
                    st.code(turn["sql"], language="sql")
            if turn.get("fig") is not None:
                st.plotly_chart(turn["fig"], use_container_width=True)
            if turn.get("df") is not None:
                st.dataframe(turn["df"], use_container_width=True)

    for turn in st.session_state.history:
        render(turn)

    question = st.chat_input("Ask a business question...") or st.session_state.pop("pending", None)
    if not question:
        return

    user_turn = {"role": "user", "text": question}
    st.session_state.history.append(user_turn)
    render(user_turn)
    st.session_state.messages.append({"role": "user", "content": question})
    ctx = {"df": None, "sql": None, "fig": None}

    with st.chat_message("assistant"):
        with st.status("Agent working...", expanded=True) as status:
            def on_step(name, args, result):
                st.write(f"🔧 **{name}** {args if name != 'get_schema' else ''}")
                if str(result).startswith("ERROR"):
                    st.write(f"⚠️ {result}")
            try:
                answer = run_agent(st.session_state.messages, ctx, on_step)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                answer = (f"Could not reach Ollama or the model failed: `{e}`\n\n"
                          f"Make sure Ollama is running and you ran `ollama pull {MODEL}`.")
                status.update(label="Error", state="error")

    ai_turn = {"role": "assistant", "text": answer, "sql": ctx["sql"],
               "fig": ctx["fig"], "df": ctx["df"]}
    st.session_state.history.append(ai_turn)
    render(ai_turn)


main()