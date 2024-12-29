import numpy as np
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import BitsAndBytesConfig, TrainingArguments
import pandas as pd
from datasets import Dataset, DatasetDict
import pdb
import re
from trl.core import LengthSampler
from vllm import LLM, SamplingParams

# relevant items in K / numbers in GT
def recall_K(GT_items, predicted_items):

    hit_count = np.isin(predicted_items, GT_items).sum()

    return hit_count/len(GT_items)

def hr_k(GT_items, predicted_items,K):

    hit_count = np.isin(predicted_items, GT_items).sum()

    return hit_count/K

    
    # Mean Reciprocal Rank (MRR)
    # 1/n *\sum(rank_i)
    # rank_i = 1/rankc

# GT_items and predicted_items are two lists
def MRR (GT_items, predicted_items):
    N = len(GT_items)
    rank = 0
    for item in len(predicted_items):
        cur_predected = predicted_items[item]
        if cur_predected in GT_items:
            rank += 1/(item+1)
    return 1/N*rank

# return the probability of correctness for each prediction, than give the MRR
def MRR_pred (correct_prob):
    correctness = correct_prob>0.4
    try:
        # Find the index of the first correct prediction (1-based index)
        first_correct = np.where(correctness == 1)[0][0] + 1
        # Compute the reciprocal rank
        return 1 / first_correct
    except IndexError:
        # If there is no correct prediction, return 0
        return 0

def get_coverage(responses, ref_answer, score_model, tokenizer):
    # return a list containing level of coverage,  [split_1 to ref, split_2 to answer, ...]
    
    assert len(responses) ==1, "only one response for the testing at a time"
    
    for full_response in responses:
        pattern = r'\(\d+\)\s'
        segments = split_points(full_response, pattern)
        
        similarity_results = []
        for i in range(len(segments)-1):
            
            response = segments[0]+' '+segments[i+1]
            
            prompt = (f"Compare the following two sentences w.r.t the diagnosis of a patient."
                    f"Please provide a numerical estimation as to whether sentence 1 share similar information to sentence 1,  ranging from 0 to 1."
                    f"Such as whether they describe similar symptoms, similar body parts, and similar diagnosis."
            f"where 0 means sentence 1 containing no information from sentence 2, and 1 means sentence 1 fully covers information from sentence 2. \n\n"
            f"Sentence 1: \"{response}\"\n"
            f"Sentence 2: \"{ref_answer}\"\n"
            f"Coverage score:")
                
            inputs = tokenizer.encode(prompt, return_tensors="pt")
                # answer = pl(f"Context: {context}\n\nQuestion: {question}\n\nAnswer: ")[0]['generated_text'].split('Answer: ')[1]
            with torch.no_grad():
                output = score_model.generate(inputs.to('cuda'),max_new_tokens=100, return_dict_in_generate=True, output_scores=True,pad_token_id=tokenizer.eos_token_id)
            
            output_sent = output.sequences[0] #[inputs.shape[1]:]
            answer = tokenizer.decode(output_sent,skip_special_tokens=True)

            pattern = r'Coverage score:\s*(-?\d+(\.\d+)?)'
            
            match = re.search(pattern, answer)
            if match:
                similarity = float(match.group(1))
            else:
                similarity = 0
            
            # print(f"similarities between {response} and {ref_answer}: {similarity}")        
            similarity_results.append(similarity)

    return similarity_results
    
def split_points(text, pattern):
    # Use re.split to separate the points
    parts = re.split(pattern, text)
    
    # Filter out empty parts and reattach the delimiter
    points = [f"({i}) {part.strip()}" for i, part in enumerate(parts) if part.strip()]

    return points
    
def truncate_after_second_delimiter(segment):
    # Define the regex pattern to match content up to the second comma or period
    pattern = r'^((?:[^,.]*[,.]){2}[^,.]*)'
    match = re.match(pattern, segment)
    if match:
        segment = match.group(1)
    
    pattern = '\n'
    if pattern in segment:
        segment = segment.split(pattern)[0]
    
        
    return segment
    
def get_coverages(responses, ref_answers, score_model, tokenizer):
    # return a list of list: [[response_1 to ref_answer 1, response_2 to ref_answer 1, ...], [response_1 to ref_answer 2, response_2 to ref_answer 2, ...],...]
    
    # split ref_answers
    pattern = r'\(\d+\)\s'
    segments = split_points(ref_answers, pattern)
    
    coverage_lists = []
    
    for i in range(len(segments)-1):
        
        coverage_lists.append(get_coverage(responses, segments[0]+' '+segments[i+1], score_model, tokenizer))
    
    return coverage_lists
    

def test_instance(tokenizer, model, query, ref_answer, gen_kwargs, score_model):
    # only takes one input instance
    with torch.no_grad():
        output = model.generate(query, gen_kwargs,
                )
        
        responses = [output[0].outputs[0].text,]
    
    cleaned_responses = []
    pattern = r'\(\d+\)\s'
    for response in responses:
        segments = split_points(response, pattern)
        truncated_segments = [truncate_after_second_delimiter(segment) for segment in segments]
        cleaned_responses.append(''.join(truncated_segments))
    
    coverage_results = get_coverages(cleaned_responses, ref_answer, score_model, tokenizer)
    
    return coverage_results

'''
tokenizer = AutoTokenizer.from_pretrained("./experiments/checkpoint_llama2Top/",device_map='auto')
if tokenizer.pad_token is None:
    # Add a new padding token to the tokenizer
    # tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained("./experiments/checkpoint_llama2Top/",device_map='auto')
'''
model_name = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
model = LLM(model=model_name,
          gpu_memory_utilization=.5, quantization="AWQ")

#'''


tokenizer = AutoTokenizer.from_pretrained('/home/tkz5084/llama7B')
score_model = AutoModelForCausalLM.from_pretrained(
        '/home/tkz5084/llama7B',
        device_map="auto", 
        quantization_config=BitsAndBytesConfig(load_in_4bit=True))

'''
with open('data/all_converted_patient_category.json', 'r') as f:
    data = json.load(f)
'''

with open('data/top_converted_patient.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_pandas(pd.DataFrame(data))
train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
dataset_dict = DatasetDict({
    'train': train_val_split['train'],
    'test': train_val_split['test']
})
test_dataset = train_val_split['test']

def format_query(example):
    example['query'] = f"### Question: {example['Context']}. What is the potential diagnosis over this patient? \n ### Answer: "
    return example

def tokenize_function(example):
    return tokenizer(example['query'], truncation=True, padding="do_not_pad",max_length = 1024)

test_dataset = test_dataset.map(format_query)
tokenized_datasets = test_dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch")


generation_config = SamplingParams(temperature=0.8, top_p=0.95)


similaritis = 0
recalls = 0
accuracies = 0
MRRs = 0
num = 0

for sample in tokenized_datasets:
    
    # collect similarities with diff
    similarity_all = test_instance(tokenizer, model, sample['query'], sample['Ans'], generation_config, score_model=score_model )#sample['category']

    # collect 
    similarity_matrix = np.array(similarity_all)
    
    if similarity_matrix.size == 0:
        print('no similarity matrix')
        continue
    
    max_similarity_array = similarity_matrix.max(axis=1)

    recall = (similarity_matrix.max(axis=1) > 0.4).sum()/similarity_matrix.shape[0]
    mean_similarity = max_similarity_array.mean()
    accuracy = (similarity_matrix.max(axis=0) > 0.4).sum()/similarity_matrix.shape[0]
    MRR_value = MRR_pred(similarity_matrix.max(axis=0))
    
    ratio = num/(num+1)
    similaritis = similaritis*ratio + mean_similarity*(1-ratio)
    recalls = recalls*ratio + recall*(1-ratio)
    accuracies = accuracies*ratio + accuracy*(1-ratio)
    MRRs = MRRs*ratio + MRR_value*(1-ratio)
    
    print(f"averaged similarity : {similaritis}, averaged recalls: {recalls}, averaged accuracy: {accuracies}, averaged MRR: {MRRs}")
    num = num+1