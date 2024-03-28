import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model = AutoModelForCausalLM.from_pretrained("distilgpt2", device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model2 = AutoModelForCausalLM.from_pretrained(r"C:\Users\Tairo Kageyama\Documents\GitHub\Python-fo-Natural-Language-Processing-main\lab8\models\V1",
                                              load_in_8bit = False)
# prompt = "How many songs have been recorded throughout history? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."



def answerQn(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    max_length = 128
    no_repeat_ngram_size = 3

    outputs = model2.generate(input_ids=input_ids, max_length=max_length, no_repeat_ngram_size=no_repeat_ngram_size)
    # print(len(outputs[0]))
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text.replace(prompt, "")
    ans = generated_text.replace(prompt, "")
    # ans = 'Hello'
    return ans

# ----------------------------------------------------------------------------------------


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Chat Bot"),
    dcc.Input(id='user_input', type='text', placeholder='Enter a question'),
    # dcc.Input(id='query', type='text', placeholder='Enter a word'),
    html.Button('Ask!!', id='process_button'),
    html.Div(id='output_div')
])

@app.callback(
    Output('output_div', 'children'),
    [Input('process_button', 'n_clicks')],
    [dash.dependencies.State('user_input', 'value')]
)
def process_word(n_clicks, user_input):
    if n_clicks is not None and n_clicks > 0:
        output = answerQn(user_input) #here
        return [
            html.P(f'You entered: {user_input}'),
            html.P(f'Answer : {output}'),
        ]
    

if __name__ == '__main__':
    app.run_server(debug=True)