# import thư viện
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes()
sns.set(style="whitegrid")
from scipy.stats import zscore
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

image = Image.open('download.jpg')

# set page
st.set_page_config(page_title="E-commerce", page_icon=":money_with_wings:")





# import thư viện Content Base
# add styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# function popularity collaborative fillering:
class popularity_based_recommender_model():
    def __init__(self, train_data, test_data, user_id, item_id):
        self.train_data = train_data
        self.test_data = test_data
        self.user_id = user_id
        self.item_id = item_id
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def fit(self):
        #Get a count of user_ids for each unique product as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)
    
        #Sort the products based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(20)

    #Use the popularity based recommender system model to make recommendations
    def recommend(self, user_id, n=5):    
        user_recommendations = self.popularity_recommendations
        
        #Filter products that are not rated by the user
        products_already_rated_by_user = self.train_data[self.train_data[self.user_id] == user_id][self.item_id]        
        user_recommendations = user_recommendations[~user_recommendations[self.item_id].isin(products_already_rated_by_user)]
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols].head(n)     
       
        return user_recommendations
    
    
    def predict_evaluate(self):        
        ratings = pd.DataFrame(self.train_data.groupby(self.item_id)['Rating'].mean())
        
        pred_ratings = [];            
        for data in self.test_data.values:
            if(data[1] in (ratings.index)):
                pred_ratings.append(ratings.loc[data[1]])
            else:
                pred_ratings.append(0)
        
        mse = mean_squared_error(self.test_data['Rating'], pred_ratings)
        rmse = sqrt(mse)
        return rmse



  
# upload data
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=True)
for file in uploaded_files:
    if file.name=="consine_similarity.npy" :
        result=np.load(file)
        
    elif file.name=="newdata.csv" :
        data=pd.read_csv(file)
    elif file.name=="reviewdata.csv" :
        data2=pd.read_csv(file)
        data2=data2[['userId','productId','Rating']]
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(data2, test_size =.20, random_state=10)

        pr = popularity_based_recommender_model(train_data=train_data, test_data=test_data, user_id='userId', item_id='productId')
        pr.fit()

# def list products for collaborative filtering        
def list_products(text):
    list_products=pr.recommend(text)
    lst=list_products['productId'].tolist()
    return lst

# Define page layout
menu = ["Home","Personalized Items[Content Based]","Reference Recommended Items[Collaborative Filtering]"]
choice = st.sidebar.selectbox("You are at ", menu)




if choice == "Home":
    # Create a container for the header
    with st.container():
        st.title("WELCOME TO EXPERIENCE RECOMMENDED ITEMS!")
        st.write("A recommendation system (or recommender system) is a class of machine learning that uses data to help predict, narrow down, and find what people are looking for among an exponentially growing number of options.")
        col1, col2, col3 = st.columns(3)
        
        # Display a sample product in each column
        with col1:
            st.write("Content filtering:")
            st.write("uses the attributes or features of an item such as item description, group price and rating level to recommend other items similar to the user’s preferences")
            st.image(image)            
            
            
        with col2:
            st.write("Collaborative filtering")
            st.write("Collaborative filtering algorithms recommend items (this is the filtering part) based on preference information from many users (this is the collaborative part). This approach uses similarity of user preference behavior,  given previous interactions between users and items, recommender algorithms learn to predict future interaction")
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQLb6qM94DoDyn6Rbz4xRys50Iq--9cDL3-zXdrKE1UCYJgsvhGSjAqb2wFg7CbWR9ixnA&usqp=CAU")
            
        
        with col3: 
            st.write("Popularity Model and Collaborative Filtering")
            st.write("Popularity based recommendation system uses the items that are in trend right now. It ranks products based on its popularity i.e. the rating count. If a product is highly rated then it is most likely to be ranked higher and hence will be recommended. As it is based on the products popularity, this can not be personalized and hence same set of products will be recommended for all the users.")
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_OvF4T5Yd66AJDDY-ioFGEtZEk9BDw64MzA&usqp=CAU")
              
elif  choice == "Personalized Items[Content Based]":
      st.sidebar.write("Bạn muốn tìm sản phẩm nào") 
      
      item=st.sidebar.text_input(' ')
      clicked= st.sidebar.button("Search")   
      clicked="Enter"             
           
      if  choice == "Personalized Items[Content Based]" and len(item)!=0:  
            with st.container():
                st.title("Một số đề xuất cho bạn tham khảo!")
                
                item.lower()
                item_index=data.loc[data['name'].str.lower().str.startswith(item)].index[0]  ##this is the problem
                distances=result[item_index]
                item_list=sorted(list(enumerate(distances)), reverse=True , key=lambda x:x[1])[1:10]
                col1, col2 = st.columns([5, 3])
        
            for i in item_list:

                if item_list.index(i)%2==0:
                    with col1:
                        st.image(data.iloc[i[0],11],use_column_width=True)
                        st.write(data.iloc[i[0],3])
                        st.write(data.iloc[i[0],8])
                        st.write(data.iloc[i[0],5])
                else: 
                    with col2:
                        st.image(data.iloc[i[0],11],use_column_width=True)
                        st.write(data.iloc[i[0],3])
                        st.write(data.iloc[i[0],8])
                        st.write(data.iloc[i[0],5])
        
            

            
elif choice == "Reference Recommended Items[Collaborative Filtering]":
    st.sidebar.write("Vui lòng nhập mã ID của bạn")      
    item=st.sidebar.text_input(' ')
    clicked= st.sidebar.button("Search")   
    clicked="Enter" 
    def list_products(lst,text):
        list_products=pr.recommend(text)
        lst=list_products['productId'].tolist()
        return lst
    
        
    
    # Create a container for the team members
    with st.container():
         st.subheader("Bạn có thể thích 1 số sản phẩm sau dựa trên Popularity Collaborative Filtering")
       
       
         
    
    with st.container():      
        list_products=pr.recommend(item)
        lst=list_products['productId'].tolist()      
        col1, col2 , col3, col4, col5= st.columns(5)
        st.write("Một số đề xuất cho bạn tham khảo!")
                        
    for item_check in lst:
                i=data.loc[data['item_id']==item_check].index[0]
                
                if lst.index(item_check)==0:
                    with col1:
                        st.image(data.iloc[i,11],use_column_width=True)
                        st.write(data.iloc[i,3])
                        st.write(data.iloc[i,8])
                        st.write(data.iloc[i,5])
                elif lst.index(item_check)==1:
                    with col2:
                        st.image(data.iloc[i,11],use_column_width=True)
                        st.write(data.iloc[i,3])
                        st.write(data.iloc[i,8])
                        st.write(data.iloc[i,5])
                elif lst.index(item_check)==2:
                    with col3:
                        st.image(data.iloc[i,11],use_column_width=True)
                        st.write(data.iloc[i,3])
                        st.write(data.iloc[i,8])
                        st.write(data.iloc[i,5])
               
                elif lst.index(item_check)==3:
                    with col4:
                        st.image(data.iloc[i,11],use_column_width=True)
                        st.write(data.iloc[i,3])
                        st.write(data.iloc[i,8])
                        st.write(data.iloc[i,5])

                else:
                    with col5:
                        st.image(data.iloc[i,11],use_column_width=True)
                        st.write(data.iloc[i,3])
                        st.write(data.iloc[i,8])
                        st.write(data.iloc[i,5])





