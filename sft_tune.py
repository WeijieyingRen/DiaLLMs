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
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl.core import LengthSampler
import pdb
import pandas as pd
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import json
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import tensorboard
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

def sft_train() -> None:
    """Train the model."""

    args = parse_args()
    
    
    pretrained_model = AutoModelForCausalLM.from_pretrained(
    args.ppo_model_name,
    device_map="auto", # 加了这两句会报错，原因是没有自定义attention_mask，导致数据格式不一致报错
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
    #load_in_8bit=True
    )
    
    # Tokenizer & dataset.
    tokenizer = AutoTokenizer.from_pretrained('/home/tkz5084/llama7B',device_map = 'auto',fast_tokenizer=True)
    if tokenizer.pad_token is None:
        # Add a new padding token to the tokenizer
        # tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.pad_token_id = 0

    '''
    with open('data/all_converted_patient_category.json', 'r') as f:
        data = json.load(f)
    '''
     
    with open('data/top_converted_patient.json', 'r') as f:
        data = json.load(f)
    
    
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    train_val_split = dataset.train_test_split(test_size=0.999, seed=42)
    dataset_dict = DatasetDict({
        'train': train_val_split['train'],
        'test': train_val_split['test']
    })
    
    
    def formatting_prompts_func(example):
        output_texts = []
        
        for i in range(len(example['Context'])):
            if len(example['Context'][i])>600:
                #pdb.set_trace()
                context = example['Context'][i][:600]
            else:
                context = example['Context'][i]
            
            text = f"### Question: {context}. \n ### Answer: {example['Ans'][i]}"
            output_texts.append(text)
        
        return output_texts

    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=25,
        report_to="tensorboard",
        num_train_epochs=50,
        max_steps=-1,
        gradient_checkpointing=False,
        )
    
    # Step 4: Define the LoraConfig
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        bias="none",
        task_type="CAUSAL_LM",
        )

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=pretrained_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_dict['train'],
        formatting_func=formatting_prompts_func,
        peft_config=peft_config,
        max_seq_length=args.max_length,
    )

    trainer.train()
    trainer.save_model(
        os.path.join(
            args.output_dir,
            "SFT_tune",
            args.run_name,
            'model',
        ),
    )


if __name__ == '__main__':
    # To train the reward model, run the following command:
    
    # To train the PPO model, run the following command:
    #ppo_args = get_ppo_parse_args()
    sft_train()