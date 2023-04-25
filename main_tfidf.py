import os
import csv
import math
###Read in all sys call traces as tuples in the 'traces' list: (type,list of sys call sequence)
traces = []
#Imports the attack system call traces to the list
atkbase = "C:/Users/grego/PycharmProjects/thesis/ADFA-LD/Attack_Data_Master/"
types = ['Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
for elem in types:
   for i in range(1,11): #upper bound of 11 for full file import
      path = atkbase + elem + "_" + str(i)
      os.chdir(path)
      def read_files(file_path):
         with open(file_path, 'r') as file:
            mystring = file.read()
            x = mystring.split()
            traces.append(('1',x))
      # Iterate over all the files in the directory
      for file in os.listdir():
         if file.endswith('.txt'):
            # Create the filepath of particular file
            file_path =f"{path}/{file}"
            read_files(file_path)
#Imports (appends) the training/test data folders' system call traces
normal = ("C:/Users/grego/PycharmProjects/thesis/ADFA-LD/Training_Data_Master/","C:/Users/grego/PycharmProjects/thesis/ADFA-LD/Validation_Data_Master/")
for elem in normal:
   os.chdir(elem)
   def read_files(file_path2):
      with open(file_path2, 'r') as file:
         mystring = file.read()
         x = mystring.split()
         traces.append(('0', x))
   for file in os.listdir():
      # Check whether file is in text format or not
      if file.endswith(".txt"):
         file_path2 = f"{elem}\{file}"
         # call read text file function
         read_files(file_path2)
num_traces = len(traces)

###Create a dictionary of features
featDict = dict()
#Desired n-gram length(s)
n=[1,2]
#Iterates through all sys call traces to build feature dictionary
for ns in n:
    for seqs in traces:
       grams = set([' '.join(seqs[1][i:i + ns]) for i in range(len(seqs[1]) + 1 - ns)])
       for gram in grams:
          if gram not in featDict:
             featDict[gram] = 1
          else:
             featDict[gram] += 1

#Generate a list of all unique terms from the dictionary (featList)
featList = ['Label']
for keys in featDict:
   featList.append(keys)
print('n-grams:',n)
print("Number of Features:",len(featList)-1)

###Create the data vectors for each system call and export to a .csv file (rows = traces)

#Function to return a dictionary of feature TF-IDF values for any given sys call trace
def feature_tfidf(listname, n, features):
    counts = dict()
    # make n-grams as string iteratively
    grams = [' '.join(listname[i:i + n]) for i in range(len(listname)+1 - n)]
    for gram in grams:
        counts[gram] = 0
    for gram in grams:
        if gram in features:
            counts[gram] += (1/len(grams)) #gives rel term freq
    #multiply each counts dict value by log(total number of traces/num traces containing term)
    for keys in counts:
        counts[keys] = (counts[keys])*math.log(num_traces/(featDict[keys]))
    return counts

print('Writing data vectors to .csv file...')
#Change working directory
os.chdir("C:/Users/grego/PycharmProjects/thesis")
#Creates a feature vector for each sys call trace and exports to .csv file in above working directory
s =''
for elem in n:
    s+=str(elem)
with open('test_data'+s+'_tfidf.csv', 'w',newline="") as csvfile:
    # writing features as header
    writer = csv.DictWriter(csvfile, fieldnames=featList, extrasaction='ignore')
    writer.writeheader()
    # Calculating values of each feature for each file
    for eachTrace in traces:
         feature_count = {}
         for elem in n:
            feature_count.update(feature_tfidf(eachTrace[1], elem, featList))
         feature_count.update({'Label': eachTrace[0]})
         for f in featList:
             if f not in feature_count:
                 feature_count.update({f: 0})
         writer.writerow(feature_count)
print('.CSV FILE COMPLETE')

