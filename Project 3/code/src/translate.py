import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_sent_cut_of_end(path):
    res_l = []
    with open(path,"r") as f:
        for l in f.readlines():
            res_l.append(l[:-1])
    return res_l
def zero_shot(sentence,model,tokenizer,device,start_lang="Persian", target_language="English", use_persian_promt=False,verbose=False):
              # Define the Zero-Shot prompt
    #sentence = 'Hello, how are you?'
    if not use_persian_promt:
        zero_shot_prompt = "You are a helpful translator. "\
            "Please translate the following "+start_lang+" sentence into "+target_language+". "\
            "Do not add any extra text or explanations. "\
            "Only provide the translation. "\
            "\n\n"\
            +start_lang+": "+sentence+"\n"\
            +target_language+":"
    else: 
        zero_shot_prompt = "تو خیلی مترجم خوبی هستی. لطفا جمله مقابل رو از فارسی به انگلیسی  ترجمه کن هیچ توضیح  اضافی نده عزیزم و هیچ متنی جز متن ترجمه ننویس.\
فارسی:"+sentence+"\
انگلیسی:"
        
    
    # Tokenize input
    inputs = tokenizer(zero_shot_prompt, return_tensors="pt").to(device)
    
    # Generate output
    with torch.no_grad():
        if use_persian_promt:
            max_new_tokens=150
        else: 
            max_new_tokens=50
            
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            eos_token_id=tokenizer.eos_token_id
        )
    if verbose:
        print("_____________")
        print("Size Output:",len(outputs[0]))
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if not gen_text[-1:].isspace():
        print("During Output The Modle Hit an Output Limit")

    if verbose:
        print("Original Output:",gen_text)
    
    # post-process output
    #if "German:" in gen_text:
     #   gen_text = gen_text.split("German:")[-1].strip()
    start_marker = target_language+":"
    gen_text_original = gen_text
    if start_marker in gen_text:
        gen_text = gen_text.split(start_marker)[-1].split("\n")[0].strip()
        if gen_text == "" or gen_text.rstrip() == "":
            #print("Solution seems to be empty Original Output was:\n" +gen_text_original)
            return False , gen_text_original
    else:
        return False , gen_text_original

    return  True , gen_text.rstrip()

def translate_multiple(sentence_l,model,tokenizer,device,start_lang="Persian", target_language="English",use_persian_promt=False,verbose=False):
    for index, sentence in enumerate(sentence_l):
        #print(type(sentence))
        #if index + 1 != 34:
        #    continue 
        has_result, result = zero_shot(sentence,model,tokenizer,device,start_lang=start_lang, target_language=target_language,use_persian_promt=use_persian_promt,verbose=verbose)

        if result: 
            print(f"[{index+1}]{sentence} \n[{index+1}] {result}")
        else: 
            print(f"[{index+1}]{sentence} \n[{index+1}] No Result Ouptut was:\n {result}")
            print("----")





