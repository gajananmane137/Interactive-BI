Your question → LLM → "call run_sql" → Python runs it → result → LLM → answer
                  ↑___________ repeats until done ___________↓
1. Imports and config
python
MODEL = "qwen2.5:7b"
MAX_STEPS = 6
MAX_ROWS_TO_LLM = 20
ollama talks to your local model, pandas handles tables, plotly draws charts, streamlit builds the web page, and sqlite3 is Python’s built-in database.
MAX_STEPS stops the agent from looping forever if the model keeps making mistakes.
MAX_ROWS_TO_LLM sends only 20 rows back to the model. Small local models get confused by big tables, and it saves time.

SYSTEM_PROMPT is the agent’s instruction manual. It tells the model the workflow: get the schema, write SQL, fix errors, make a chart, then answer using only the data. The last line about dates helps it write correct SQLite date queries, because SQLite has no real date type.

2. Sample data and database setup

generate_sample_csv(path) builds 1,200 fake orders. random.seed(42) makes the data identical on every run. For each order it picks a random date, customer, region, product and discount, then calculates sales and profit. Orders with a 20% discount lose extra profit, which is a hidden pattern for the agent to find. It then writes everything to a CSV.

setup_database() runs once:

python
if os.path.exists(DB_PATH): return      # already built, skip
if not os.path.exists(CSV_PATH): generate_sample_csv(CSV_PATH)
df = pd.read_csv(CSV_PATH)
df.to_sql(TABLE, con, ...)              # CSV → SQL table

So the flow is CSV → pandas → SQLite file sales.db. This is why you need no account.

connect_readonly() opens the database with mode=ro (read-only). Even if the model writes DROP TABLE, SQLite itself refuses. This is the first safety layer.

FORBIDDEN is a regex that catches dangerous words such as delete, update and drop. It’s the second safety layer.

3. The tools

These are normal Python functions the model can request. Each one takes ctx, a small dictionary ({"df", "sql", "fig"}) that holds the latest result for the current question so the UI can show it later.

get_schema returns the CREATE TABLE statement plus 3 sample rows. The model needs to know column names and what the values look like (for example, that region contains “North” and not “north”).

run_sql has several steps:

It checks the query starts with SELECT or WITH and contains no forbidden words.
It runs the query with pd.read_sql_query.
If it fails, it returns "ERROR: ..." as text instead of crashing. The model reads this and fixes its SQL, which is the self-correction loop and the “agentic” part.
If it succeeds, it saves the dataframe in ctx and returns the row count, column names and the first 20 rows as CSV text.

make_chart checks that a query was run and that x and y are real column names. It then builds a bar, line, pie or scatter chart with Plotly and stores it in ctx["fig"]. The model never sees the picture. It only gets “Chart created” back.

4. Tool definitions (TOOLS)
python
{"type": "function", "function": {
    "name": "run_sql",
    "description": "Run a read-only SQLite SELECT query...",
    "parameters": {...}}}

This is a JSON description of each tool: name, purpose and parameters. Ollama passes it to the model, which is how the model learns what it can call. Good descriptions matter, because the model decides which tool to use based on them. The enum on kind limits the chart type to four valid choices.

REGISTRY maps the tool name (a string) to the real Python function, so the code can look up "run_sql" and call it.

5. The agent loop (run_agent)

This is the core of the app.

python
for _ in range(MAX_STEPS):
    resp = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
    msg = resp.message
    messages.append(msg)
    if not msg.tool_calls:
        return msg.content          # model gave a final answer → done
    for call in msg.tool_calls:
        result = fn(ctx, **args)    # run the tool
        messages.append({"role": "tool", ...})   # give result back

Each loop works like this:

Send the whole conversation to the model.
If the model replies with plain text, that’s the final answer and the loop stops.
If the model replies with tool_calls, run each tool in Python, add the results to the conversation as role: "tool" messages, and loop again.

A typical run is: get_schema → run_sql → make_chart → final text. If SQL fails, you get an extra run_sql round.

on_step is a callback that lets the UI show live progress (“🔧 run_sql ...”). The try/except TypeError handles the case where the model calls a tool with wrong argument names.

6. The UI (main())

Setup: setup_database() runs first. Then st.session_state stores two things that survive Streamlit’s page reruns:

messages: the full conversation the LLM sees, which gives it memory for follow-ups.
history: what is displayed on screen (text, SQL, chart and table).

Sidebar: the example buttons set st.session_state.pending, which acts as if you typed that question. “Clear chat” resets both lists.

render(turn): draws one chat bubble with the text, an expandable “SQL used” section, the chart and the table.

Handling a question:

python
question = st.chat_input(...) or st.session_state.pop("pending", None)

It takes either typed input or a clicked example. Then it:

shows your message and appends it to messages
creates a fresh ctx
calls run_agent inside st.status(...), which shows the live “Agent working...” box
catches errors such as Ollama not running and shows a helpful message
saves the answer, SQL, chart and table into history and renders them
7. Why it’s built this way
Design choice	Reason
Errors returned as text, not raised	Lets the model fix its own mistakes
Read-only DB + keyword filter	Two safety layers against destructive SQL
Only 20 rows sent to the LLM	Saves context, and small models cope better
MAX_STEPS	Prevents infinite loops
ctx dictionary	The UI gets the full dataframe and chart, not just the model’s summary
Full messages kept in session	Follow-up questions like “now by month” work
