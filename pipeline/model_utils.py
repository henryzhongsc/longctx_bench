import logging
import os
import json
import pdb

logger = logging.getLogger("main")


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, chat_template):

    if chat_template is None:
        return prompt
        
    if "llama3" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in chat_template or "vicuna" in chat_template:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "recurrentgemma" in chat_template:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "mamba-chat" in chat_template:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "rwkv" in chat_template:
        prompt =  f"""User: hi

                Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

                User: {prompt}

                Assistant:"""
    else:
        logger.error(f"{chat_template} is unsupported.")
        raise NotImplementedError

    return prompt
