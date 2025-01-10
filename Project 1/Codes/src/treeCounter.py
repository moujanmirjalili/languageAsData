from collections import OrderedDict 
from src.helper import printTree
from src.helper import getDepthAndDegree
from src.helper import avDict
from src.helper import printDict
from src.helper import get_lines_without_number

from collections import Counter
import statistics
def get_label(w):
    return w.dep_

#Counts different properties for a Spacy tree
#whole_text :: the whole text as list of strings with every string one sentence 
#nlp :: the trained nlp 
#tree_to_check :: (root,List_of_childreen])
#Gives a tree of max two depth that schould be checked if in the structure
# returns ==>
# av_depth:: The average depth 
#mean_degree_dict :: a dictionary with keys labels and values the average degree of these labels in the text 
#mean_distance_dict :: a dictionary with keys labels and values the average distance to the root 
#leave_counter :: A counter object with counted how often a label is a leave.
#count_ancestors :: A dictionary with keys labels and values counter objects often a label was a ancestor this is also the structure for the next parameters. Ancestors are every node above the node including the root 
#count_descendants :: same as above but counts descendants which is the complete subtree 
#count_head :: same as above but counts heads which is just the direct predecessor of a node 
#count_childreen :: same as above but only counts childreen which are the direct child nodes 
#all_sent_with_structure :: all sentences which contains the tree_to_check as a subtree
def tree_counter(whole_text,nlp,tree_to_check):

    
    root_to_check, child_to_check = tree_to_check
    
    depth_sum = 0
    count = 0
    all_sent_with_structure = []
    completeText = len(whole_text)
    count_degree_dict = {}
    count_distance_dict = {}
    leave_counter = Counter()
    count_ancestors = {}
    count_descendants = {}
    
    count_head = {}
    count_childreen = {}
    for index, sent in  enumerate(whole_text):
        tree = nlp(sent)
        single_sent = next(tree.sents)
        # Calls the tree traversal function 
        single_depth, degree_tupel_l = getDepthAndDegree(single_sent.root, currentDis = 0,isRoot = True)
        depth_sum += single_depth
        for label, degree, distance, isLeave in degree_tupel_l:
            if isLeave:
                leave_counter.update([label])
            if label not in count_degree_dict : 
                count_degree_dict[label] = [degree]
                count_distance_dict[label] = [distance]
            else: 
                count_degree_dict[label] += [degree]
                count_distance_dict[label] += [distance]
        count += 1 
        already_added_sent = False
        for word in single_sent:
            label = word.dep_
            childreen = list(map(get_label,word.children))
            head = word.head.dep_

            
            subtree = list(map(get_label,word.subtree))
            ancestors = list(map(get_label,word.ancestors))#list(word.ancestors) # subtree
            if label in count_ancestors:
                count_ancestors[label].update(ancestors)
                count_descendants[label].update(subtree)
                count_head[label].update([head])
                count_childreen[label].update(childreen)
                
            else: 
                count_ancestors[label] = Counter(ancestors)
                count_descendants[label] = Counter(subtree)
                count_head[label] = Counter([head])
                count_childreen[label] = Counter(childreen)
            if word.dep_ == root_to_check and not already_added_sent:
                is_sub_tree = True
                for check_c in child_to_check:  
                    if not check_c in childreen:
                        is_sub_tree = False
                        break
                if is_sub_tree:
                    all_sent_with_structure.append(single_sent)
                    already_added_sent = True

                        
            #head = word.head
            #subtree = list(word.subtree)
            
            
            
        if index%50 == 0:
            print('\r', 'Progress: ',index,'/',completeText, end='')
        #if index == 220:
        #    break
        #displacy.render(tree, style="dep", jupyter=True, options={"distance": 90})
        #break
    av_depth = depth_sum/count
    mean_degree_dict  = avDict(count_degree_dict)
    mean_distance_dict = avDict(count_distance_dict)


    return (av_depth,mean_degree_dict,mean_distance_dict,leave_counter,count_ancestors,count_descendants,count_head,count_childreen,all_sent_with_structure)
    print("\nTotal sentences: ",count," Average Depth: ",depth_sum/count)




