import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

dataframe = pd.read_csv('diabetes.csv')

print(dataframe.head(5))
print(dataframe.shape)


diabetes_data = dataframe.copy(deep=True)
# diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

histogram_data = diabetes_data.hist(figsize = (20,20))
print(histogram_data)
# sns.pairplot(diabetes_data,hue='Outcome')
# plt.show()

sns.FacetGrid(diabetes_data, hue="Outcome", size=5) \
   .map(sns.distplot, "BMI") \
   .add_legend()
plt.show()



# import missingno    #BAr Graphs for Data Visualisation
# p = missingno.bar(dataframe)
# plt.show()

# p=pd.plotting.scatter_matrix(dataframe,figsize=(25, 25))
# plt.show()



#HEAT MAPS
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(diabetes_data.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
plt.show()

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_data.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])

print(X.head)

# X = X.drop('Outcome',axis = 1)
Y = diabetes_data.Outcome



from sklearn.model_selection import train_test_split
X_Train , X_Test, Y_Train  , Y_Test = train_test_split(X,Y,test_size = 1/3,random_state=42,stratify = Y)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

test_score = []
train_score = []

model = GaussianNB()
# for i in range(1,15):
#    knn = KNeighborsClassifier(i)
#    knn.fit(X_Train,Y_Train)
gaussian = model.fit(X_Train,Y_Train)
train_score.append(gaussian.score(X_Train,Y_Train))
test_score.append(gaussian.score(X_Test,Y_Test))

print("Test Score is ",test_score)
print("Train Score is ",train_score)





from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = gaussian.predict(X_Test)
confusion_matrix(Y_Test,y_pred)
print(pd.crosstab(Y_Test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

