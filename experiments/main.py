'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from absl import app
from ml_collections import config_flags
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM)

from llm_attacks import get_goals_and_targets, get_workers

_CONFIG = config_flags.DEFINE_config_file('config')

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def get_ids(worker,goal,target,model):
    output = []
    # print(model)
    for i in range(len(goal)):
        #prompt and input
        prompt = "[INST] " + goal[i] + " " + target[i]
        encoded_prompt = worker.tokenizer.encode(prompt)
        decoded_prompt = worker.tokenizer.decode(encoded_prompt)
        # print("SAVING PROMPT", decoded_prompt)

        # print('prompt is:',prompt)
        inputs = worker.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attn_masks = torch.ones_like(input_ids).to(model.device)

        # combined = worker.tokenizer.encode("[INST] " + goal[i], )
        # combined.extend(worker.tokenizer.encode(target[i])[1:])
        # for j, (id1, id2) in enumerate(zip(input_ids[0].tolist(), combined)):
        #     if id1 != id2:
        #         print("FAILURE AT", j)
        #         print(input_ids[0][:j+1].tolist())
        #         print(worker.tokenizer.decode(input_ids[0][:j+1].tolist()))
        #         print("\"" + worker.tokenizer.decode([id1]) + "\"")
        #         print(combined[:j+1])
        #         print(worker.tokenizer.decode(combined[:j+1]))
        #         print("\"" + worker.tokenizer.decode([id2]) + "\"")
        #         # print("EOS TOKEN ID", worker.tokenizer.eos_token_id)
        #         break

        goal_length = len(worker.tokenizer.encode("[INST] " + goal[i]))
        target_length = len(worker.tokenizer.encode(target[i]))
        # print('input length is',input_length,' goal length is',goal_length,' target length is', target_length)
        # get the logits
        logit = model(input_ids,attn_masks).logits
        # print('logit shape is (before)',logit.shape)
        logit = logit[:, goal_length - 1 : goal_length + target_length + 16, :]
        # target_token_ids = torch.argmax(logit, dim=-1)
        # decoded_sentence = worker.tokenizer.batch_decode(target_token_ids, skip_special_tokens=True)[0]
        # print('output string is:',decoded_sentence)
        # tokens = worker.tokenizer.convert_ids_to_tokens(target_token_ids[0].tolist())
        # print('target token is',tokens)
        # print('logit shape is (after)',logit.shape)
        # probability = F.softmax(logit,dim = -1)
        # print('the probability shape is',probability.shape)
        output.append(logit)

    # put logits to the file
    torch.save(output,'ids.pth')
    # a = torch.load('ids.pth')

    # print('the shape is',[i.shape for i in a])
    return 0

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value
    params.devices = ['cuda:1']

    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')

    print(params)

    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)

    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
    test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]

    workers, test_workers = get_workers(params)

    get_ids(workers[0],train_goals,train_targets,workers[0].model)

    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    else:
        attack = attack_lib.IndividualPromptAttack(
            train_goals,
            train_targets,
            workers,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=getattr(params, 'test_goals', []),
            test_targets=getattr(params, 'test_targets', []),
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    app.run(main)