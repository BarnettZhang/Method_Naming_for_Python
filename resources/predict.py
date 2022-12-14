from distutils.command.config import config
from transformers import RobertaTokenizer
import torch
import dill
import sys
from time import time
import os.path
import zipfile
from CodeBERTaModel import CodeBERTaEncoderDecoder
from Config import Config

def predict(src_path, selected_code, device):
    model_name = 'QuantizeState'
    
    # Check if model is unzipped, if not, unzip the model
    if not os.path.isfile(src_path + model_name):
        with zipfile.ZipFile(src_path + model_name + '.zip', 'r') as zip_ref:
            zip_ref.extractall(src_path)
        os.remove(src_path + model_name + '.zip')

    # Load the model to CPU
    if device == 'false': 
        device = torch.device("cpu")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    model = CodeBERTaEncoderDecoder(config,device)
    model.load_state_dict(torch.load(src_path + model_name, pickle_module=dill, map_location='cpu'), strict=False)

    # Tokenize the selected code and make the prediction using the model. 
    # Adapted from the final project of Ugo Benassayag, https://github.com/UgoBena/Sourcery_Project
    selected_code_raw = selected_code
    tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
    selected_code = tokenizer.batch_encode_plus([selected_code])["input_ids"]
    underscore_token = tokenizer.get_vocab()["_"]
    selected_code = [[token for token in input if token != underscore_token][:tokenizer.model_max_length] for input in selected_code]
    evaluate_input = [(selected_code[0], [0, 883, 19603, 8588, 2], len(selected_code[0]), 5, selected_code_raw[0])]
    top_sequence,top_length,tgt_seqs,tgt_lens,output_prob,decoded_sequences,inputs_raw = model.evaluate(evaluate_input)

    return decoded_sequences

if __name__ == "__main__" :
    # Print the prediction output, extension.js will collect the stdout and display it on the msg.
    name_list = predict(sys.argv[1], sys.argv[2], sys.argv[3])
    ind = 1
    for name in name_list[0]:
        print(str(ind) + '. ' + name.lstrip("_").rstrip("_"))
        ind = ind + 1
    sys.stdout.flush()