from dff.script import GLOBAL, TRANSITIONS, RESPONSE, Context, Actor, Message
import dff.script.conditions.std_conditions as cnd
from typing import Union
import re

def hi_lower_case_condition(ctx: Context, actor: Actor, *args, **kwargs) -> bool:
    request = ctx.last_request
    # Returns True if `hi` in both uppercase and lowercase
    # letters is contained in the user request.
    if request is None or request.text is None:
        return False
    return "hi" in request.text.lower()


def dude_condition(ctx: Context, actor: Actor, *args, **kwargs) -> bool:
    request = ctx.last_request
    return bool(re.search(r'dude|bro|man', request.text))


def not_(f):
    def n_f(*args, **kwargs):
        return not f(*args, **kwargs)
    
    return n_f
# create a dialog script
script = {
    "greeting_flow": {
        "start_node": {  # This is the initial node,
            # it doesn't contain a `RESPONSE`.
            RESPONSE: Message(),
            TRANSITIONS: {
                # "node1": cnd.exact_match(Message(text="Hi"))
                "dude": dude_condition,
                'start_node': not_(dude_condition),
            },
            # If "Hi" == request of user then we make the transition
        },
        "dude": {
            RESPONSE: Message(text="Hi, dude, how are you?"),
            TRANSITIONS: {
                "node2": cnd.regexp(r".*how are you", re.IGNORECASE),
                'dude': not_(cnd.regexp(r".*how are you", re.IGNORECASE))
            },
            # pattern matching (precompiled)
        },
        "node2": {
            RESPONSE: Message(text="Good. What do you want to talk about?"),
            TRANSITIONS: {"node3": cnd.all([cnd.regexp(r"talk"), cnd.regexp(r"about.*music")])},
            # Mix sequence of condtions by `cnd.all`.
            # `all` is alias `aggregate` with
            # `aggregate_func` == `all`.
        },
        "node3": {
            RESPONSE: Message(text="Sorry, I can not talk about music now."),
            TRANSITIONS: {"start_node": cnd.regexp(re.compile(r"Ok, goodbye."))},
            # pattern matching by precompiled pattern
        },
        "fallback_node": {  # We get to this node
            # if an error occurred while the agent was running.
            RESPONSE: Message(text="Ooops"),
            TRANSITIONS: {
                "dude": dude_condition,
                # The user request can be more than just a string.
                # First we will check returned value of
                # `complex_user_answer_condition`.
                # If the value is `True` then we will go to `node1`.
                # If the value is `False` then we will check a result of
                # `predetermined_condition(True)` for `fallback_node`.
                "fallback_node": cnd.true(),  # or you can use `cnd.true()`
                # Last condition function will return
                # `true` and will repeat `fallback_node`
                # if `complex_user_answer_condition` return `false`.
            },
        },
    }
}

# init actor
actor = Actor(script, start_label=("greeting_flow", "start_node"))


# handler requests
def turn_handler(in_request: Message, ctx: Union[Context, dict], actor: Actor):
    # Context.cast - gets an object type of [Context, str, dict] returns an object type of Context
    ctx = Context.cast(ctx)
    # Add in current context a next request of user
    ctx.add_request(in_request)
    # Pass the context into actor and it returns updated context with actor response
    ctx = actor(ctx)
    # Get last actor response from the context
    out_response = ctx.last_response
    # The next condition branching needs for testing
    return out_response, ctx


ctx = {}

print('I\'m DSM bot')
while True:
    in_request = input("type your answer: ")
    out_response, ctx = turn_handler(Message(text=in_request), ctx, actor)
    print(out_response.text)