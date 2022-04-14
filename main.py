import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


#SEE 'README' FILE for very brief info about the organization used




#1) loading internal dataset of seaborn
titanic = sns.load_dataset('titanic')

# print(titanic.head())
# titanic.info()
# sns.set_style('whitegrid')
# sns.set_context('paper')

#the data is consisted 15 columns, we intend to predict the 'survived'
#column




#2) >>>>>>>>>>>>>>>>>>DATA ANALYSIS SECTION>>>>>>>>>>>>>>>>>>>>>>>>>
#lets explore some of the columns and how they relate
#this is a classification problem, there's no one way to explore data


#importantly, we look at the column we want to to predict
#we see survived and non-survived

# sns.countplot(x='survived',data=titanic,palette='coolwarm')
# #we have more non-survivors than survivors
# plt.show()


#we may look at jointplots of some columns maybe related

# sns.jointplot('fare','age',data=titanic)
# #tha fare and age column are loosely-related from the plot
# plt.show()


#we may also choose to look at distribution of columns like 'fare' column

# sns.distplot(titanic['fare'],bins=15,color='r')
# #most fares are in the range of 0-50 from this plot
# plt.show()


#we print the correlation between terms to see good-corrrelations

# print(titanic.corr())
#we can see correlation on the survived column below
# print(titanic.corr()['survived'].sort_values(ascending=False))


#We can also visualize these corrrelations in a heatmap

# sns.heatmap(titanic.corr(),annot=True,cmap='coolwarm')
# #the 'fare' column has some positive corr with the 'survived' colm
# #while the 'adult_male' column has a negative correlation with it
# plt.show()


#note that extra arguments like: palette, cmap, color, bin etc would
#not stop the figures from being plotted, theyre used to provide more
#detail to these figures


#SLIGHTLY ADVANCED ACTIVITY

#in some cases like here we have few columns, we can use a pairplot
#to see all columns and do a quick data analysis instantly
#BUT FIRST, all boolean columns like 'adult_male' & 'alone' have to be
#mapped to some dummy variables so they can participate in the plot

#mapping for boolean columns -> integer

titanic['adult_male']=titanic['adult_male'].map({True:1,False:0})
titanic['alone']=titanic['alone'].map({True:1,False:0})

#also for string columns i interested in like the 'sex' & 'alive' columns
titanic['sex']=titanic['sex'].map({'male':1,'female':0})
titanic['alive']=titanic['alive'].map({'yes':1,'no':0})
# titanic.info()


#SECOND, is to work on the 'age' column & fill the missing ages
#you can also drop the missing entries
#I chose to fill the missing ages with the average age
avg = titanic['age'].mean()  #-> the average age
titanic['age'] = titanic['age'].apply(lambda x: avg if np.isnan(x) else x)

#the pairplot for the columns

# sns.pairplot(titanic)
# #Note: that a lot of the data you may get in the pairplot like may
# #have no actual meaning;tho they can quickly reveal good-correlations
# plt.tight_layout()
# plt.show()
# #note that columns that contain 'strings' are not used in pairplots

#>>>>>>>>>>>>>>>>>>END OF DATA ANALYSIS SECTN>>>>>>>>>>>>>>>>>>>>>>





#3) fitting a logistic regression model on this data
#this will be done in two steps
#first decide how to handle 'string columns'
#then secondly create a logistic model & train the numerical colmns

#so, we will drop all string columns main reason is to keep this simple

#FirstPart, these columns we drop:
#['embarked', 'class', 'who', 'deck', 'embark_town']

#dropping these columns
titanic.drop(['embarked', 'class', 'who', 'deck', 'embark_town'],axis=1,inplace=True)

#to quickly see how the new data nowlooks
# print(titanic.head())


#SecondPart, to create the model and fit on the remaining data

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#the seconD part goes in very short stepS

#seperate data into input and outputs
inputs = titanic.drop('survived',axis=1)
output = titanic['survived']

#------------------------------------------------------------------
#OPTIONAL! quickly checking info on the data check no missing entry
# inputs.info()
# output.info()
#------------------------------------------------------------------


#split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs,output,test_size=0.3,random_state=101)


#------------------------------------------------------------------
#OPTIONAL! create a scaler object and scale/transform all inputs only
#note use fit_transform only on 'X_train'

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#------------------------------------------------------------------


#CREATE YOUR LOGISTIC MODEL AND TRAIN THE MODEL

model = LogisticRegression()

#FIT/TRAIN on the training data-pair
model.fit(X_train,y_train)

#THEN, make predictions & EVALUATE on the testing data pair
predictions = model.predict(X_test)


#for classification tasks like this one, we can use confusionmatrix
#and classificationreport to see how we performed

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions),'\n')


#-------------ENDS HERE-------------------------------------------

# print(titanic.groupby('survived')['survived'].count())





#>>>>>>>>>>>>>>>>>>>>>>>>OPTIONAL>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
# #----------SLIGHTLY ADVANCED OPTIONAL PART-------------------------
#
# #model performed great got 100% test classification;
#
#
# #WE MAY MAKE A RANDOM PREDICTION FROM THE INITIAL DATASET
#
# #say we select passenger no 8; who actually didnt survive -> [0]
# print(titanic['survived'][7],'\n')
#
# #lets predict what our model says about this person
# #nB: if used scaling?, remember to scale input before applying
#
# #getting input before scaling
# test_input = titanic.iloc[7].drop('survived')
#
# #scaled_input
# input_sc = scaler.transform([test_input])
# #show model_prediction about passenger 8
# print(model.predict(input_sc))
#
# #MODEL ALSO PREDICTED CORRECTLY THAT THIS PASSENGER; DID NOT SURVIVE
# # #---------------------------------------------------------------
#
# #EXPLANATION ON OPTIONAL PART ABOVE
# #used .iloc[] to get the eighth-row (index starts from 0), then
# #dropped the 'survived' column since thats what we will predict
# #the resulting data becomes our input, we scale this input and pass
# #the scaled-data into the model to make predictions
#
#>>>>>>>>>>>>>>>>>>>>>>>>OPTIONAL>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


#visit scikit-learn documentation page on using linear_models and
#logistic regression models
#feel free to search the internet on "use of different functions"
#that may look new while reading through

