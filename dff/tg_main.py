import os
import re

import torch
import pickle
from dff.script import conditions as cnd
from dff.script import labels as lbl
from dff.script import RESPONSE, TRANSITIONS, Message
from dff.messengers.telegram import PollingTelegramInterface
from dff.pipeline import Pipeline
from dff.script import GLOBAL, TRANSITIONS, RESPONSE, Context, Actor, Message
from dff.utils.testing.common import is_interactive_mode

from model import TransformerModel, generate, sent_from_seq, device, PositionalEncoding
from torchtext.data.utils import get_tokenizer


model = torch.load('model.pt')
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
tokenizer = get_tokenizer('basic_english')


def generated_response(ctx: Context, actor: Actor, *args, **kwargs) -> Message:

    start_tokens = torch.tensor([torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in ctx.last_request.text.split(' ')]).to(device)
    seq = generate(model, start_tokens, 10)
    generated_text = sent_from_seq(seq, vocab)
    return Message(text=generated_text)

def yes_condition(ctx: Context, actor: Actor, *args, **kwargs) -> bool:
    request = ctx.last_request
    return bool(re.search(r'yes|y|of course|ofk', request.text))

def no_condition(ctx: Context, actor: Actor, *args, **kwargs) -> bool:
    request = ctx.last_request
    return bool(re.search(r'no|n|don\'t', request.text))

def idk_condition(ctx: Context, actor: Actor, *args, **kwargs):
    return not yes_condition(ctx, actor, *args, **kwargs) and not no_condition(ctx, actor, *args, **kwargs)


default_trans = {
    ('greeting_flow',"greeting_node"): cnd.exact_match(Message(text="/start")),
}

script = {
    "greeting_flow": {
        "start_node": {
            TRANSITIONS: {**default_trans},
        },
        "greeting_node": {
            RESPONSE: Message(text="Hi, wanna generate? (y/n) the model just continues whatever you write in exact 10 words!"),
            TRANSITIONS: {
                **default_trans,
                ('generating_flow', 'gen_node'): yes_condition,
                'no_node': no_condition,
                'idk_node': idk_condition
            },
        },
        'no_node': {
            RESPONSE: Message(text='Unfortunately, I\'m too stupid to do anything else...'),
            TRANSITIONS: {
                **default_trans,
                'start_node': cnd.true()
            }
        },
        'idk_node':{
            RESPONSE: Message(text='I haven\'t clearly understood. Repeat please, but as slowly, as you can...'),
            TRANSITIONS: {
                **default_trans,
                ('generating_flow', 'gen_node'): yes_condition,
                'no_node': no_condition,
                'idk_node': idk_condition
            }
        },
        "fallback_node": {
            RESPONSE: Message(text="Please, repeat the request"),
            TRANSITIONS: {**default_trans},
        },
    },
    'generating_flow': {
        'gen_node': {
            RESPONSE: generated_response,
            TRANSITIONS: {
                **default_trans,
                lbl.repeat(): cnd.true()
            }
        }
    }
}

# this variable is only for testing
happy_path = (
    (Message(text="/start"), Message(text="Hi")),
    (Message(text="Hi"), Message(text="Hi")),
    (Message(text="Bye"), Message(text="Hi")),
)

interface = PollingTelegramInterface(token=os.getenv("TG_BOT_TOKEN", ""))

pipeline = Pipeline.from_script(
    script=script,  # Actor script object
    start_label=("greeting_flow", "start_node"),
    fallback_label=("greeting_flow", "fallback_node"),
    messenger_interface=interface,  # The interface can be passed as a pipeline argument.
)


def main():
    if not os.getenv("TG_BOT_TOKEN"):
        print("`TG_BOT_TOKEN` variable needs to be set to use TelegramInterface.")
    pipeline.run()


if __name__ == "__main__" and is_interactive_mode():  # prevent run during doc building
    main()