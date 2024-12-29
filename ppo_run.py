import os
import time

import torch
import argparse
from trainer.args import parse_args
from trainer.data import collator
from trainer.data import get_tokenizer
from trainer.data import load_data
from trainer.reward import reward_fn, reward_consist, reward_sentTrans
from trainer.ppo_trainer import build_trainer
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl.core import LengthSampler
import pdb
import pandas as pd
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import json

"""
def get_ppo_parse_args():
    parser = argparse.ArgumentParser(description="hyperparameter for ppo model")

    parser.add_argument('--peft_lora_r', type=int, default= 2 , help='peft_lora_r')
    parser.add_argument('--peft_lora_alpha', type=int, default= 16 , help='peft_lora_alpha')
    parser.add_argument('--ppo_reward_model_name', type=str, default='/home/tkz5084/llama7B', help='ppo_reward_model_name')
    parser.add_argument('--ppo_model_name', type=str, default='/home/tkz5084/llama7B', help='ppo_model_name')
    parser.add_argument('--steps', type=int, default= 100000 , help='peft_lora_r')
    parser.add_argument('--learning_rate', type=int, default= 1e-5, help='peft_lora_alpha')
    parser.add_argument('--log_with', type=str, default='tensorboard', help='')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--mini_batch_size', type=int, default=1, help='')
    
    parser.add_argument('--optimize_cuda_cache',  type=bool, default= True , help='peft_lora_r')
    parser.add_argument('--early_stopping', type=bool, default= False , help='peft_lora_alpha')
    parser.add_argument('--target_kl', type=float, default=0.1, help='')
    parser.add_argument('--ppo_epochs', type=int, default=50, help='')
    parser.add_argument('--init_kl_coef', type=float, default=0.2, help='')
    
    parser.add_argument('--adap_kl_ctrl', type=bool, default= True , help='peft_lora_r')
    #parser.add_argument('--tracker_project_name', type=int, default= 16 , help='peft_lora_alpha')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--adafactor', type=bool, default=False, help='')
    
    args = parser.parse_args()
    return args
"""

def create_prompt(sentence1, sentence2):
    if len(sentence1) > 1024:
        sentence1 = sentence1[:1024]
    if len(sentence2) > 1024:
        sentence2 = sentence2[:1024]
         
    return (f"Compare the following two sentences w.r.t the diagnosis of a patient and provide a numerical estimation of their consistency in terms of meaning on a scale from 0 to 10, "
        f"where 0 means completely different and 10 means exactly the same meaning.\n\n"
        f"Sentence 1: \"{sentence1}\"\n"
        f"Sentence 2: \"{sentence2}\"\n"
        f"Consistency score:")

def ppo_train() -> None:
    """Train the model."""

    args = parse_args()
    
    # Tokenizer & dataset.
    tokenizer = AutoTokenizer.from_pretrained('/home/tkz5084/llama7B',device_map = 'auto',fast_tokenizer=True)
    
    if tokenizer.pad_token is None:
        # Add a new padding token to the tokenizer
        # tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.pad_token_id = 0

    with open('data/all_converted_patient_category.json', 'r') as f:
        data = json.load(f)
       
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
    dataset_dict = DatasetDict({
        'train': train_val_split['train'],
        'test': train_val_split['test']
    })
    def tokenize_function(example):
        return tokenizer(example["Context"], truncation=True, padding="do_not_pad",max_length = 1024)
        # example['Ans'], example['category']
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch")
    

    # PPO Trainer.
    _, ppo_trainer = build_trainer(
        args = args,
        tokenizer=tokenizer,
        dataset=tokenized_datasets['train'],
        data_collator=collator,
    )

    gen_kwargs = {
        'top_k': 0.0,
        'top_p': 0.9,
        'do_sample': True,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }
    output_length_sampler = LengthSampler(
        30,
        100,
    )

    # Reward model.
    #reward_tokenizer = AutoTokenizer.from_pretrained(
    #    args.reward_model_name,
    #    device_map = 'auto',
    #    fast_tokenizer=True
    #)

    '''
    reward_model = AutoModelForCausalLM.from_pretrained(
        '/home/tkz5084/llama7B',
        device_map="auto", 
        quantization_config=BitsAndBytesConfig(load_in_4bit=True)
    )
    reward_tokenizer = tokenizer
    '''
    reward_model2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    

    # ppo training
    #ppo_trainer.accelerator.print(f'Using device: {device}')
    start_time = time.time()
    # Training loop.
    for epoch in tqdm(range(args.ppo_epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            #prompt_tensors = batch['input_ids']
            prompt_tensors = batch['input_ids']
            response_tensors = ppo_trainer.generate(
                query_tensor = prompt_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **gen_kwargs,
            )
            batch['response'] = tokenizer.batch_decode(
                response_tensors,
                skip_special_tokens=True,
            )
            print(batch['response'])
            # what is the format of the function? 
            # Compute reward score.
            '''
            scores = reward_consist(
                model=reward_model,
                tokenizer=reward_tokenizer,
                prompt_func=create_prompt,
                response_text=batch['response'],
                reference_text = batch['Ans'],
                device='cuda',
            )
            '''
            
            scores = reward_sentTrans(
                model=reward_model2,
                response_text=batch['response'],
                reference_text = batch['Ans'],
                device='cuda',
            )
            
            rewards = [
                torch.tensor(score - args.reward_baseline)
                for score in scores
            ]
            
            # Run the PPO step.
            stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        # Save the model.
        if args.save_freq and epoch % args.save_freq == 0:
            ppo_trainer.save_pretrained(
                os.path.join(
                    args.output_dir,
                    args.project_name,
                    args.run_name,
                    f'checkpoint-{epoch}',
                ),
            )

    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    hours, mins = divmod(mins, 60)
    ppo_trainer.accelerator.print(f'Training took {hours:.0f}h {mins:.0f}m {secs:.0f}s.')

    ppo_trainer.accelerator.print('\nSaving model!')
    ppo_trainer.save_pretrained(
        os.path.join(
            args.output_dir,
            args.project_name,
            args.run_name,
            'model',
        ),
    )


if __name__ == '__main__':
    # To train the reward model, run the following command:
    
    # To train the PPO model, run the following command:
    #ppo_args = get_ppo_parse_args()
    ppo_train()