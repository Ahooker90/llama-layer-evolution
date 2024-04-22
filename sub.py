import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon as JSD

def list_argmax(iterable):
    #https://stackoverflow.com/questions/16945518/finding-the-index-of-the-value-which-is-the-min-or-max-in-python
    return max(enumerate(iterable), key=lambda x: x[1])[0]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("/work/ree398/nlp/ass_1c/nlp_1c/Llama-2-7b-hf_output_dir", output_hidden_states=True)
model.eval()

input_text = "What is the capital of Washington? Answer:"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states

data_dict = {
    'hidden_states':[], #This is the collection of hidden states collected during a forward pass of our architecture
    'layer_outputs':[], #This is the collection of all layers passed through the models final layer 4096->32000 (vocab size)
    'probs':[], #This is the collection of layer_outputs being passed through a softmax to create a probability distribution
    'token_ids':[], #This is the collection of probs being passed through an argmax(-1), giving us a collection of the most probable token from the probability distribution
    'last_token_distributions':[]#This is the collection of distributions for ONLY the last token (used for predicting the next work during generation)
    }

for hidden_state in hidden_states:
    layer_output = model.lm_head(hidden_state)
    prob = F.softmax(layer_output, dim=-1)
    token_id = prob.argmax(-1)
    data_dict['hidden_states'].append(hidden_state)
    data_dict['layer_outputs'].append(layer_output)
    data_dict['probs'].append(softmax(model.lm_head(hidden_state),dim=-1))
    data_dict['token_ids'].append(token_id)
    
num_bins =32000
top_k = 5
layer_names = [f"Layer {i}" for i in range(len(data_dict['layer_outputs']))]
if True:#plots the distribution for the final token prediction (i.e. exclude 0:n-1)
    for i, probs in enumerate(data_dict['probs']):
        last_ind =data_dict['probs'][0].shape[1]-1 #n-1
        last_distribution = probs[0,last_ind,:]
        data_dict['last_token_distributions'].append(last_distribution)
        top_values, top_indices = torch.topk(last_distribution , top_k)
        top_tokens = [tokenizer.decode([top_indices[i]]) for i in range(top_k)]
        if True:
            plt.figure(figsize=(12, 6))
            plt.bar(top_tokens, top_values)
            plt.title(f'Top {top_k} Predictions from {layer_names[i]}')
            plt.ylabel('Probability')
            plt.xlabel('Tokens')
            plt.xticks(rotation=45)
            plt.show()
"""
----------Factuality-----------
We observe the calculated JSD would be still extremely high in the higher layers. 
This pattern indicates that the model is still changing its predictions in the last few layers, 
and potentially injecting more factual knowledge into the predictions.
citation: [https://openreview.net/pdf?id=Th6NyL07na]

https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
"""
distance_measure_between_probability_array = []
for i in range(len(data_dict['last_token_distributions'])-1):
    p = data_dict['last_token_distributions'][i]
    q = data_dict['last_token_distributions'][i+1]
    distance_measure_between_probability_array.append(JSD(p, q))
plt.figure(figsize=(12, 6))
plt.plot(distance_measure_between_probability_array)
plt.title('Jessen Shannon Distance Between Layers')
plt.ylabel('Distance')
plt.xlabel('Layer Number')
plt.show()
print(f'Most Factual Layer: {list_argmax(distance_measure_between_probability_array)}')




"""
#Used for development and later exploratory processes

#Grab the outputs from layer 10 and 32 [batch_size, num_of_tokens, layer_size=4096]
early_layer_output = hidden_states[8]
final_layer_output = hidden_states[32] 

#Pass the layer outputs through the final linear layer to project embeddings (layer_ouputs) into the vocabulary space
early_exit_outputs = model.lm_head(early_layer_output)
final_exit_outputs = model.lm_head(final_layer_output)

#Transform final-linear-layer-projection-onto-vocab into a probability distribution
early_exit_probs = F.softmax(early_exit_outputs, dim=-1)
final_exit_probs = F.softmax(final_exit_outputs, dim=-1)

early_exit_tok_id = early_exit_probs.argmax(-1)
final_exit_tok_id = final_exit_probs.argmax(-1)
"""


"""
#Exploratory Processes
if False: # This shows the prediction for each token at layer 8 and 32(hidden_states[])
    for i in range(6):
        top_values, top_indices = torch.topk(early_exit_probs[0,i,:] , top_k)
        top_tokens = [tokenizer.decode([top_indices[i]]) for i in range(top_k)]
        plt.figure(figsize=(12, 6))
        plt.bar(top_tokens, top_values)
        plt.title(f'Early Layer Position {i}')
        plt.ylabel('Probability')
        plt.xlabel('Tokens')
        plt.xticks(rotation=45)
        plt.show()
    
    for i in range(6):
        top_values, top_indices = torch.topk(final_exit_probs[0,i,:] , top_k)
        top_tokens = [tokenizer.decode([top_indices[i]]) for i in range(top_k)]
        plt.figure(figsize=(12, 6))
        plt.bar(top_tokens, top_values)
        plt.title(f'Final Layer Position {i}')
        plt.ylabel('Probability')
        plt.xlabel('Tokens')
        plt.xticks(rotation=45)
        plt.show()


if False: #print the raw probabilities no filtering
    for i in range(6):
        plt.figure(figsize=(12, 6))
        plt.plot(final_exit_probs[0,i,:])
        plt.title('test')
        plt.ylabel('Probability')
        plt.xlabel('Tokens')
        plt.xticks(rotation=45)
        plt.show() 

if False:#plots the first top_k number of tokens and shows the token prediction
    for i, probs in enumerate(data_dict['probs']):
        # Assuming each layer's output is [batch_size, seq_length, vocab_size] and we're focusing on the first token
        #top_values, top_indices = torch.topk(probs , top_k) #https://pytorch.org/docs/stable/generated/torch.topk.html
        top_indices = probs.argmax(-1).squeeze(0)
        top_probabilities = [probs[0,i,top_indices[i]] for i in range(top_k)]
        top_tokens = [tokenizer.decode([top_indices[i]]) for i in range(top_k)]
        plt.figure(figsize=(12, 6))
        plt.bar(top_tokens, top_probabilities)
        plt.title(f'Top {top_k} Predictions from {layer_names[i]}')
        plt.ylabel('Probability')
        plt.xlabel('Tokens')
        plt.xticks(rotation=45)
        plt.show()
"""





























