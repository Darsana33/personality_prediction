import pandas as pd
import numpy as np
from extractor1 import extract_text_from_doc
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
#from pyresparser import ResumeParser
import pickle

traindata =pd.read_csv('train_dataset.csv')
array = traindata.values

df=pd.DataFrame(array)
maindf =df[[1,2,3,4,5,6]]
mainarray=maindf.values
#print(mainarray)

temp=df[7]
train_y =temp.values
# print(train_y)
# print(mainarray)

for i in range(len(train_y)):
	train_y[i] =str(train_y[i])

classifier = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
classifier.fit(mainarray, train_y)

with open('classifier.pk1', 'wb') as pickle_file:
        pickle.dump(classifier , pickle_file)

with open('classifier.pk1', 'rb') as pickle_file:
        classifierpk = pickle.load(pickle_file)
		
		
dir="input\\input.docx"

text_data=extract_text_from_doc(dir)
text_data=text_data.replace(',',"")
text_data=text_data.lower()
text_data=text_data.split(" ")
#print(text_data)
#print(len(text_data))

'''
print(text_data[0])
for i in range(len(text_data)):
	print(text_data[i])
'''

opnss=['imaginative','insightful','curious','creative','outspoken','straightforward','direct','receptive','open-minded','adventurous']
cons=['thoughtful','goal-oriented','ambitious','organised','mindful','vigilant','control','disciplined','reliable','responsible']
extr=['cheerful','sociable','talkative','assertive','outgoing','energetic','extroverted','friendly','enthusiastic','outspoken']
agree=['trustworthy','altruism','kind','affectionate','cooperative','empathetic','modest','sympathetic','compliant','tender-minded']
neuro=['calm','strong hearted','collected','balanced','peaceful','tranquil','strong-willed','stable']

opnss_val=0
cons_val=0
extr_val=0
agree_val=0
neuro_val=0
temp=0
age=0

for i in text_data:
	temp=temp+1
	if i=='age':
		age=int(text_data[temp])
	if i in opnss:
		if opnss_val<10:
			opnss_val=opnss_val+1
	if i in cons:
		if cons_val<10:
			cons_val=cons_val+1
	if i in extr:
		if extr_val<10:
			extr_val=extr_val+1
	if i in agree:
		if agree_val<10:
			agree_val=agree_val+1
	if i in neuro:
		if neuro_val<10:
			neuro_val=neuro_val+1
	
#print("age:",age)	
#print(opnss_val)
#print(cons_val)
#print(extr_val)
#print(agree_val)
#print(neuro_val)

		
testdata =pd.read_csv('test.csv')
arr_test = testdata.values
df_test=pd.DataFrame(arr_test)
df_test[[1]]=age
df_test[[2]]=opnss_val
df_test[[3]]=neuro_val
df_test[[4]]=cons_val
df_test[[5]]=agree_val
df_test[[6]]=extr_val
arr_test=df_test[[1,2,3,4,5,6]]
#print(arr_test)    
#arr_test=
	   
#y_pred = classifierpk.predict([[18,7,6,3,7,1]])
y_pred = classifierpk.predict(arr_test)
print("\nPersonality is ",y_pred[0],".")

f=open("out.txt","w")
f.write(y_pred[0])
f.close()


