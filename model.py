
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from groq import Groq


# Loading dataset
data = pd.read_csv("salaries.csv")


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Kaggle Data Insights with LLM"),
    html.Div("Ask anything about the Kaggle data!"),
    dcc.Textarea(id="user-input", placeholder="Ask a question about the data...", style={'width': '100%', 'height': 100}),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='response-output', style={'whiteSpace': 'pre-line'})
])

# Function to send a query to Groq LLM using Groq library
def query_groq_llm(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-8b-8192",  # This is an example model name. Modify it as per the model you want to use.
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {e}"

# Callback to handle user input and generate LLM response
@app.callback(
    Output('response-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('user-input', 'value')
)
def generate_response(n_clicks, user_input):
    if n_clicks > 0 and user_input:
        # Query Groq LLM API
        prompt = f"Using the following data:\n{data.head().to_string()}\nAnswer the following question: {user_input}"
        response = query_groq_llm(prompt)
        return response
    return ""

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
