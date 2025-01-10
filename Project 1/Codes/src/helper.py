import re
import statistics



regex_clean = r"[^0-9a-zA-Z\sáéíóúüñ]"
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

def clean_spanish_text(whole_text):
    temp = []
    for lin in whole_text:
        t = re.sub(regex_clean, '', lin)
        t2 = re.sub("\n", '.', t)
        #print(".")
        temp.append(t2)
        #temp += t2
    return temp


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










