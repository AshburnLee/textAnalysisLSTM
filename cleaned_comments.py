##############################################
# output the csv stores cleand comment 
##############################################
from nltk.corpus import stopwords
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import sys

'''
provide train.csv & test.csv
'''


train_path = sys.argv[1]
test_path = sys.argv[2]



'''
read dataset, 
and get the 'comment_text' column,
create a stop_words instance, which is a set()
'''
#stemmer = SnowballStemmer('english')  # did not use stemmer
stop_words = set(stopwords.words("english"))   # set()

# read raw file:  -> DataFrame
train_set = pd.read_csv(train_path)
#train_set.head()
test_set = pd.read_csv(test_path)
#test_set.head()

# get the 'comment column':
#train_text = train_set["comment_text"]
#test_text = test_set["comment_text"]
train_text = train_set.loc[:,"comment_text"]
test_text = test_set.loc[:,"comment_text"]


'''
get the labels from training set: y,
get each comment from all dataset,
'''

# get the comment in ndarrays and clean it:
# Get labels arrays & train array:
labels_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
get_labels = train_set[labels_list]  #-> DataFrame
y = get_labels.values  #-> ndarray -> (159571,6)


train_array = train_set["comment_text"].values  # -> ndarray  (159571,)
test_array = test_set["comment_text"].values   # -> ndarray  (153164,)




'''
create a fucntion to clean all comments,
create two empty lists.
'''
# A list to store cleaned text
train_cleaned_comment_list = [] 
test_cleaned_comment_list = []


# set one peise of comment as para
def clean_comment(comment):
    # remove anything that are not words
    re_train = str(re.sub("[^a-zA-Z]"," ", comment))
    #print(retrain)
    
    # 
    tokened_train = word_tokenize(re_train.lower())
    #print(tokened_train)
    
    # remove stop_words that have not contribution to the meaning
    stoped_train = [_ for _ in tokened_train if _ not in stop_words]
    #print(stoped_train)
    
    # connect words as a string
    stoped_train_str = " ".join(stoped_train) # -> string
    
    return stoped_train_str





'''
Input cleaned data into lists and files
'''
# write to filess
count = 0
my_file = open('./cleaned_train_comment.csv','w')
for _ in tqdm(train_array):
    i = clean_comment(_)
    #print(i)
    train_cleaned_comment_list.append(i)
    my_file.write(i)
    my_file.write('\n')
    count+=1
my_file.close()
print("The length of the list(train): ",len(train_cleaned_comment_list))
print("The number of commnents(train): ", count)


count = 0
my_file = open('./cleaned_test_comment.csv','w')
for _ in tqdm(test_array):
    i = clean_comment(_)
    #print(i)
    test_cleaned_comment_list.append(i)
    my_file.write(i)
    my_file.write('\n')
    count+=1
my_file.close()
print("The length of the list(test): ",len(test_cleaned_comment_list))
print("The number of commnents(test): ", count)







