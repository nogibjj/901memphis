# """
# Main cli or app entry point
# """

# from mylib.calculator import add
# import click


# @click.command("add")
# @click.argument("a", type=int)
# @click.argument("b", type=int)
# def add_cli(a, b):
#     click.echo(add(a, b))


# if __name__ == "__main__":
#     # pylint: disable=no-value-for-parameter
#     add_cli()
from transformers import pipeline
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

generator = pipeline('text-generation', model='gpt2')

app = FastAPI()


class Body(BaseModel):
    text: str


@app.get('/')
def root():
    return HTMLResponse("<h1>A self-documenting API to interact with a GPT2 model and generate text</h1>")


@app.post('/generate')
def predict(body: Body):
    results = generator(body.text, max_length=35, num_return_sequences=1)
    return results[0]