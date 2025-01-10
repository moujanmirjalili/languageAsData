import re
import statistics
import json 
import os 

regex_clean = r"[^0-9a-zA-Z\sáéíóúüñ]"
new_reg_span = r"[^0-9a-zA-Z\sáéíóúüñ,“”.():'-»«\"]"


pers_regex1 = r'[^\u0600-\u06FF0-9\s]+'
pers_regex2 = r',|،'
pers_regex3 = r"^\d+\s*"
pers_seperator = ['\xad', '\u200e', '\u200f', '\u200d', '\u200c', '\n']

pers_remove =  r'[^\u0600-\u06FF]{3,}$' #r'^[^\u0600-\u06FF]{1,}$' 

def get_label(w):
    return w.dep_

def printTree(root):
    print(root)
    for x in root.children:
        printTree(x)

def getDepth(root):
    if root.n_lefts + root.n_rights  == 0:
        return 1
    max_sub = 0
    for x in root.children:
        dep = getDepth(x)
        if dep> max_sub:
            max_sub = dep
    return max_sub + 1
#Recurciv function used to extract information from a tree
# for first call it schould get a root 
# later the variable root will not be the root from the whole tree 
# for fist call call with currentDis = 0
#Return Value:
# (MaxDepth of the whole Graph, List of Tuples with tuple of form 
#         => (lable of entry, degreeOfNode, Distance to root, isLeaveNode)
def getDepthAndDegree(root,currentDis,isRoot=False):
    if root.n_lefts + root.n_rights  == 0:
        degree_tupel_l = [(root.dep_,1,currentDis,True)]
        return (1,degree_tupel_l)
    max_sub = 0
    count_degree = 0 if isRoot else 1
    degree_tupel_l = []
    
    for x in root.children:
        count_degree += 1
        dep , degree_tupel_l_return = getDepthAndDegree(x,currentDis=currentDis+1)
        degree_tupel_l += degree_tupel_l_return
        if dep> max_sub:
            max_sub = dep
    degree_tupel_l.append((root.dep_,count_degree,currentDis,False))
    return (max_sub + 1,degree_tupel_l)


def avDict(d):
    meanDict = {}
    for key in d:
        meanDict[key] = statistics.mean(d[key])
    
    meanDict = dict(sorted(meanDict.items(), key=lambda item: -item[1]))
    return meanDict

def printDict(d):
    for k in d:
        print(f"{k}=>{d[k]:.4f}")
    

def get_lines_without_number(file_name):
    with open(file_name, "r", encoding='utf-8') as f:
        final_L = []
        #Remove the line numbers bevor the tab
        for lin in f.readlines():
            split_lin = lin.split("\t")
            final_L.append(split_lin[1])
        return final_L

   
# Spanish ----------------

def clean_spanish_text(whole_text):
    return clean_text_replace(whole_text,regex_clean)

def clean_text_spanish_remove(whole_text):
    return clean_text_remove(whole_text,new_reg_span)
def clean_text_spanish_both(whole_text):
    return clean_text_both(whole_text,new_reg_span, regex_clean)

# Persian -----------------------------------------------

#whole_text is here a array of lines of the text 
# Return value has the same format 
 
def pers_generic_preprocessing(whole_text):
    result_l = []
    count = 0 
    for lin in whole_text:
        lin_old = lin
        for sep in pers_seperator:
            lin = lin.replace(sep, " ")
        if lin_old != lin:
            count += 1 
    
        result_l.append(lin)
    print("In: ", count," lines seperators replaced")
    return result_l

def clean_pers_text_replace(whole_text):
    temp = pers_generic_preprocessing(whole_text)
    temp = clean_text_replace(whole_text,pers_regex1)
    temp = clean_text_replace(temp,pers_regex2)
    return clean_text_replace(temp,pers_regex3)


def clean_pers_remove(whole_text):
    #pers_remove
    temp = pers_generic_preprocessing(whole_text)
    temp = clean_text_remove(temp,pers_remove)

    temp = clean_text_replace(temp,pers_regex2)
    temp = clean_text_replace(temp,pers_regex3)
    return temp
    
def clean_text_pers_both(whole_text):
    temp = clean_pers_remove(whole_text)
    temp = clean_pers_text_replace(temp)
    return temp

# --------------------------------------
def clean_text_both(whole_text,reg1,reg2):
    temp = clean_text_remove(whole_text,reg1)
    return clean_text_replace(temp,reg2)

#Cleans a text by removing every sent with unknown caracters
def clean_text_remove(whole_text,reg):
    temp = []
    pattern = re.compile(reg)
    counter = 0
    for lin in whole_text:
        if not pattern.search(lin):
            counter += 1 
            t2 = re.sub("\n", '. ', lin)
            temp.append(t2)
    print("Total lines not removed : ",counter)
    return temp 

def clean_text_replace(whole_text,reg):
    temp = []
    counter = 0 
    for lin in whole_text:
        t = re.sub(reg, '', lin)
        if t != lin:
            counter += 1 
        #    print(t)
        #    print(lin)
        #    break
        t2 = re.sub("\n", '. ', t)
        #print(".")
        temp.append(t2)
        #temp += t2
    print("Total lines replaced",counter)
    return temp

def get_cleaned_text(path,clean_func):
    whole_text = get_lines_without_number(path)
    whole_text = clean_func(whole_text)
    result_string = ""
    for x in whole_text: result_string += x
    return result_string

#Uses replace with empty 
def get_cleaned_spanish_text_as_string(path):
    return get_cleaned_text(path,clean_spanish_text)

def get_average_word_length(whole_text):
    words_len = 0
    len_words_char_total = 0
    for sentence in whole_text:
        words = sentence.split()
        words_len += len(words)
        for word in words:
            len_words_char_total += len(word)
    return len_words_char_total/words_len


def get_root_of_subtree(single_sent,tree_to_check):
    root_to_check, child_to_check = tree_to_check
    for word in single_sent:
        if word.dep_ == root_to_check:
            is_sub_tree = True
            childreen = list(map(get_label,word.children))
            for check_c in child_to_check:  
                if not check_c in childreen:
                    is_sub_tree = False
                    break
            if is_sub_tree:
                return word

def clean_persian_text(text):
    temp = []
    for lin in text:
        t1 = re.sub(r'[^ا-ی0-9\s]+', '', lin)
        #t2 = re.sub("\n", '.', t)
        #print(".")
        temp.append(t1)
        #temp += t2
    return temp

def print_all_senteces_with_structure_indication(all_sent_with_structure,tree_to_check,nlp):
    root_to_check, child_to_check = tree_to_check
    for sent in all_sent_with_structure:
        doc = nlp(str(sent))
        single_sent = next(doc.sents)
        root = get_root_of_subtree(single_sent,tree_to_check)
        if root:
            print("The sentences is: ")
            print(single_sent)
            print("Root: ",root,"Root Type: ",root.dep_)
            for child in list(root.children):
                print(child," : ",child.dep_)
        print()
        print("------------------")
        print()


def write_list_to_file(label,list_to_save,path_to_save_folder):
    complete_path = os.path.join(path_to_save_folder,label+".json")
    with open(complete_path, 'w') as config_file:
        json.dump(list_to_save, config_file)

def read_list_from_file(label,path_to_save_folder):
    complete_path = os.path.join(path_to_save_folder,label+".json")
    with open(complete_path, 'r') as config_file:   
        loadedList = json.load(config_file)
    return loadedList





