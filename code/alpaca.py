import transformers
# import sys;sys.path.append("code/DeepSpeed")
import deepspeed
import torch
import numpy as np
class Alpaca:
    def __init__(self,args):
        self.template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

        self.template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""        
        self.args = args
        self.batch_size = args.alpaca_batchsize
        self.alpaca_model, self.alpaca_tokenizer, self.ds_engine = self.get_model()
        self.instruction = None
        self.verbalizer = None
        self.num_labels = None
        self.preprocess_input = None

    def get_model(self):
        alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("alpaca/alpaca_ckpt")
        alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("alpaca/alpaca_ckpt")
        alpaca_tokenizer.padding_side = "left" 
        alpaca_model.eval()
        ds_engine = deepspeed.init_inference(alpaca_model,
                                    mp_size=self.args.world_size,
                                    dtype=torch.half,
                                    replace_method="auto",
                                    replace_with_kernel_inject=True)
        return alpaca_model, alpaca_tokenizer, ds_engine
    def inference_sample(self, Instruction,Input=None,true_input=None):

        if true_input is None:
            prompt = self.template.format(Instruction, Input)
        else:
            prompt = true_input


        inputs = self.alpaca_tokenizer(prompt, return_tensors="pt")
        # Generate
        generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),max_new_tokens=100)
        output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # if rank == 0:
        #     print('='*20)
        #     print(output)
        #     print('='*20)
    def general_preprocess_input(self,a,b):
        return a

    def as_sst2(self):
        mask_word = self.args.mask_word
        # denoise_instruction = """Replace each masked position "{mask_word}" in the provided sentence with a suitable word to make it natural and coherent. Only one word should be used to replace each "{mask_word}". The returned sentence should be of the same length as the given sentence. Provide the answer directly."""
        self.denoise_instruction = f"""Replace each mask word {mask_word} in the input sentence with a suitable word. The output sentence should be natural and coherent and should be of the same length as the given sentence.

### Input: 
{mask_word} reassembled from {mask_word} cutting-room {mask_word} of any {mask_word} daytime {mask_word} .

### Response:
apparently reassembled from the cutting-room floor of any given daytime soap .

### Input: 
a {mask_word} , funny and {mask_word} transporting re-imagining {mask_word} {mask_word} and the beast and 1930s {mask_word} films

### Response:
a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films"""

        self.instruction = """Given an English sentence input, determine its sentiment as positive or negative."""
#         self.instruction = """Given an English sentence input, determine its sentiment as "Positive" or "Negative". You can only output "Positive" or "Negative".

# ### Input: 
# apparently reassembled from the cutting-room floor of any given daytime soap .

# ### Response:
# Positive

# ### Input: 
# a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films

# ### Response:
# Negative"""

        self.verbalizer = self.sst2_verbalizer
        self.num_labels = 2
        self.preprocess_input = self.general_preprocess_input

    def sst2_verbalizer(self,output):
        if "Negative" in output or "negative" in output:
            return 0
        elif "Positive" in output or "positive" in output:
            return 1
        else:
            print('wrong: ',output)
            
            return np.random.choice([0,1])
        
    def as_agnews(self):
        mask_word = self.args.mask_word
#         self.denoise_instruciton = f"""Replace each mask word {mask_word} in the input sentence with a suitable word. The output sentence should be natural and coherent and should be of the same length as the given sentence.

# ### Input: 
# Title: Fears for T N {mask_word} after talks\nDescription: {mask_word} representing {mask_word} at Turner   Newall say {mask_word} are 'disappointed' after {mask_word} {mask_word} stricken parent firm {mask_word} Mogul.

# ### Response:
# Title: Fears for T N pension after talks\nDescription: Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.

# ### Input: 
# Title: The {mask_word} is On: Second Private Team {mask_word} Launch Date for {mask_word} Spaceflight (SPACE.com)\nDescription: {mask_word} - TORONTO, Canada -- {mask_word} {mask_word} of rocketeers competing for the  #36;10 million Ansari X Prize, a contest for\privately {mask_word} suborbital {mask_word} {mask_word} , has officially {mask_word} the first\launch date for its manned rocket.

# ### Response:
# Title: The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com)\nDescription: SPACE.com - TORONTO, Canada -- A second\team of rocketeers competing for the  #36;10 million Ansari X Prize, a contest for\privately funded suborbital space flight, has officially announced the first\launch date for its manned rocket."""
        self.denoise_instruciton = f"""f"Replace each masked position \"{mask_word}\" in the provided sentence with a suitable word to make it natural and coherent. Only one word should be used to replace each \"{mask_word}\". The returned sentence should be of the same length as the given sentence. Provide the answer directly."""


        self.instruction = """Given a news article title and description, classify it into one of the four categories: Science, Sports, Business, or World. Return the category name as the answer.

### Input: 
Title: Venezuelans Vote Early in Referendum on Chavez Rule (Reuters)
Description: Reuters - Venezuelans turned out early and in large numbers on Sunday to vote in a historic referendum that will either remove left-wing President Hugo Chavez from office or give him a new mandate to govern for the next two years.

### Response:
World

### Input:
Title: Phelps, Thorpe Advance in 200 Freestyle (AP)
Description: AP - Michael Phelps took care of qualifying for the Olympic 200-meter freestyle semifinals Sunday, and then found out he had been added to the American team for the evening's 400 freestyle relay final. Phelps' rivals Ian Thorpe and Pieter van den Hoogenband and teammate Klete Keller were faster than the teenager in the 200 free preliminaries.

### Response:
Sports

### Input:
Title: Wall St. Bears Claw Back Into the Black (Reuters)
Description: Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.

### Response:
Business
        
### Input:
Title: 'Madden,' 'ESPN' Football Score in Different Ways (Reuters)
Description: Reuters - Was absenteeism a little high\on Tuesday among the guys at the office? EA Sports would like to think it was because "Madden NFL 2005" came out that day, and some fans of the football simulation are rabid enough to take a sick day to play it.

### Response:
Science"""
        self.verbalizer = self.agnews_verbalizer
        self.num_labels = 4
        self.preprocess_input = self.general_preprocess_input

    def agnews_verbalizer(self,output):
        if "World" in output or "world" in output or "1" in output or "Politics" in output or "politics" in output:
            return 0
        elif "Sports" in output or "sports" in output or "sport" in output or "Sport" in output or "2" in output:
            return 1
        elif "Business" in output or "business" in output or "3" in output or "Finance" in output or "finance" in output:
            return 2
        elif "Science/Technology" in output or "science" in output.lower() or "technology" in output.lower() or "sci" in output.lower() or "tech" in output.lower() or "4" in output:
            return 3
        else:
            print('wrong: ',output)
            
            return np.random.choice([0,1,2,3])
        
    def denoise_instances(self,instances):
        
        denoise_instruction = self.denoise_instruciton
        text_a_list = []
        text_b_list = []
        output_list_a = []
        output_list_b = []

        for instance in instances:
            text_a_list.append(instance.text_a)
            text_b_list.append(instance.text_b)

        num = 0
        prompt_list = []
        for Input in text_a_list:
            num+=1


            # Input = "<mask> is a <mask> about a <mask> that <mask> <mask> <mask>."
            # prompt = prompt.format(Instruction,Input)
            if Input is None:
                output_list_a.append(None)
                continue
            # print()
            prompt = self.template.format(denoise_instruction, Input)

            prompt_list.append(prompt)
            if num%self.batch_size == 0:
                inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                
                # Generate
                generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),penalty_alpha=0.6, top_k=4,max_new_tokens=80)

                output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
                output = [o[len(p):] for o,p in zip(output,prompt_list)]
                # print(output)
                output_list_a.extend(output)
                prompt_list = []
        if len(prompt_list) > 0:
            inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
            
            # Generate
            generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),penalty_alpha=0.6, top_k=4,max_new_tokens=80)

            output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
            output = [o[len(p):] for o,p in zip(output,prompt_list)]
            # print(output)
            output_list_a.extend(output)
            prompt_list = []

        for output, instance in zip(output_list_a, instances):
            # print(output)
            instance.text_a = output

        for Input in text_b_list:
            num+=1


            # Input = "<mask> is a <mask> about a <mask> that <mask> <mask> <mask>."
            # prompt = prompt.format(Instruction,Input)
            if Input is None:
                output_list_b.append(None)
                continue
            # print()
            prompt = self.template.format(denoise_instruction, Input)

            prompt_list.append(prompt)
            if num%self.batch_size == 0:
                inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                
                # Generate
                generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),penalty_alpha=0.6, top_k=4,max_new_tokens=80)

                output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
                output = [o[len(p):] for o,p in zip(output,prompt_list)]
                # print(output)
                output_list_b.extend(output)
                prompt_list = []
        if len(prompt_list) > 0:
            inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
            
            # Generate
            generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),penalty_alpha=0.6, top_k=4,max_new_tokens=80)

            output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
            output = [o[len(p):] for o,p in zip(output,prompt_list)]
            # print(output)
            output_list_b.extend(output)
            prompt_list = []

        for output, instance in zip(output_list_b, instances):
            instance.text_b = output

    def predict_batch(self, instances, past_predictions):
        answers = np.zeros((len(instances),self.num_labels))
        output_list = []

        if past_predictions == None:
            text_a_list = []
            text_b_list = []

            for instance in instances:
                text_a_list.append(instance.text_a)
                text_b_list.append(instance.text_b)

            num = 0
            prompt_list = []
            for a,b in zip(text_a_list,text_b_list):
                num+=1
                Input = self.preprocess_input(a,b)

                prompt = self.template.format(self.instruction, Input)
                prompt_list.append(prompt)

                if num%self.batch_size == 0:
                    inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                    
                    # Generate
                    generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),penalty_alpha=0.6, top_k=4,max_new_tokens=80)

                    output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
                    output = [o[len(p):] for o,p in zip(output,prompt_list)]
                    # print(output)
                    output_list.extend(output)
                    prompt_list = []
            if len(prompt_list) > 0:
                inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                
                # Generate
                generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),penalty_alpha=0.6, top_k=4,max_new_tokens=80)

                output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
                output = [o[len(p):] for o,p in zip(output,prompt_list)]
                # print(output)
                output_list.extend(output)
                prompt_list = []
        else:
            output_list = past_predictions
                

        for output, answer in zip(output_list, answers):
            # print(output)
            answer[self.verbalizer(output)] = 1
            
        return answers, output_list
