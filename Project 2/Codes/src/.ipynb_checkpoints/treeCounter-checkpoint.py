from collections import OrderedDict 
from src.helper import printTree
from src.helper import getDepthAndDegree
from src.helper import avDict
from src.helper import printDict
from src.helper import get_lines_without_number

from collections import Counter
import statistics

def tree_counter(whole_text,nlp):
    def get_label(w):
        return w.dep_
    depth_sum = 0
    count = 0
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


    return (av_depth,mean_degree_dict,mean_distance_dict,leave_counter,count_ancestors,count_descendants,count_head,count_childreen)
    print("\nTotal sentences: ",count," Average Depth: ",depth_sum/count)