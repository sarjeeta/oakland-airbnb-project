import streamlit as st
import pandas as pd
import base64
import math
from PIL import Image
import xgboost as xgb
from xgboost import XGBRegressor    


from pred import *


X_whole = pd.read_csv('Oakland_final.csv')
imp_header = 'gym,dishwasher,kitchen,indoor_fireplace,iron,washer,heating,hair_dryer,stove,hot_water,microwave,dryer,Entire Home/Apartment,Private,Shared,Hotel,neighbourhood_Near North Side,neighbourhood_Albany Park,neighbourhood_Loop,neighbourhood_Lincoln Park,neighbourhood_Washington Park,neighbourhood_Lake View,neighbourhood_Logan Square,neighbourhood_Near West Side,neighbourhood_East Garfield Park,neighbourhood_South Chicago,neighbourhood_Uptown,neighbourhood_Belmont Cragin,neighbourhood_Oakland,neighbourhood_Englewood,neighbourhood_New City,neighbourhood_Austin,neighbourhood_Dunning,neighbourhood_Lincoln Highlands,room_type_Entire home/apt,room_type_Private room,neighbourhood_Woodland,neighbourhood_Upper Rockridge,air_conditioning,coffee_maker,neighbourhood_Toler Heights,neighbourhood_Highland,neighbourhood_Cleveland Heights,neighbourhood_Maxwell Park,neighbourhood_Bushrod,neighbourhood_Hoover-Foster,neighbourhood_Piedmont Avenue,refrigerator,breakfast,pool,neighbourhood_Shafter,neighbourhood_Golden Gate,neighbourhood_Downtown,neighbourhood_Lakeshore,neighbourhood_Laurel,neighbourhood_Dimond,neighbourhood_East Peralta,private_entrance'
header = imp_header.split(',')
data = []
X = pd.DataFrame(data, columns=header)

for word in header:
    data.append(0)
X.loc[1] = data


main_bg = "download"
main_bg_ext = "download"





st.title("Oakland Airbnb Data Analysis")

st.sidebar.header("CONTENTS")
box = st.sidebar.selectbox(" ", ["Project Overview", "EDA", "Feature Engineering", "Model Building", "Price Prediction"])

if box == "Project Overview":
    # st.markdown(
    # f"""
    # <style>
    # .reportview-container {{
    #     background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    # }}

    # </style>
    # """,
    # unsafe_allow_html=True
    # )
    
    st.header("An exploratory data analysis and price prediction project")
    html_temp = """
    <div style="background-color:#FFE6A0 ;padding:10px">
    <ul>
         <li> <h3> Analysing key features and metrics from house listing data </h3> </li>
         <li> <h3> Feature extraction using statistical modelling and NLP techniques </h3> </li>
         <li> <h3> Building the most suitable model for price prediction task of a house listing </h3> </li>
     </ul>
    <br>
    <p> <h3> Airbnb's publically available data of a large number of listings is a critical aspect for the company. This data is further cleaned and analyzed to provide various insights for the beneficiaries of the company.
    Various purposes include security decisions, business decisions, optimizing peer-to-peer compatibility, improving marketing and branding etc. Let us take a look into some of the visualizations that can be inferred from this dataset.
    We will try to understand the features that drive certain user decisions. Doing so, we shall subsequently try to build a model that predicts the price of an Airbnb listing.
    We will try different methods and evaluate each, thus finding the most accurate among them all for scaling into a real-world use. </h3></p>
    <br>
    <ul>
         <li> <h3> Extracting more RELEVANT data can be a future work we can look into. One such feature which guides room booking decisions strongly are the pictures of the properties listed. We can train a suitably built neural network to study such image data and make price prediction more accurate. </h3> </li>
         <li> <h3> We can also try to analyse reviews and descriptions further in detail to generate a sentiment-price link for predictions. </h3> </li>
         <li> <h3> With the availablity of a constant data flow, we can do a time-series analysis to accurately gauge geographical and seasonal dependance of bookings with the price. </h3> </li>
     </ul>
    
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    
    st.markdown("#")
    html_temp_2 = """
    <div style="background-color:#FFE6A0 ;padding:10px">
    <p> <h3> Dataset procured from <a href= "http://insideairbnb.com/get-the-data.html" target="_blank"> http://insideairbnb.com/get-the-data.html </a>  </h3></p>
    <br>
    <p> <h3> You can view the solutions python notebook <a href="https://colab.research.google.com/drive/1FL1ZW0hPtTUbL70j6WuA3uEC5_Wu54Mi" target="_blank">HERE </a> </h3> </p>
    
    """
    st.markdown(html_temp_2, unsafe_allow_html=True)






if box == "EDA":
    
    st.header("Useful metrics and insights")
    html_5 = """
    <div style="background-color:#FFE6A0 ;padding:10px">
    <p> <h3> Publically available airbnb data has to be preprocessed before automating processes to analyze it and derive insights. We begin by assesing missing values,
    dropping and formatting unwanted features, and formatting feature values to suitable datatypes. After this we can begin with the actual analysis.</h3></p>
    <br>    
    """
    st.markdown(html_5, unsafe_allow_html=True)
    st.header("Listings")
    st.write(X_whole)
    st.markdown("""<br>""", unsafe_allow_html=True)
    html_7 = """
    Since we aren't given neighbourhood groups in the dataset, let us take a look at the different unique neighbourhoods and listing concentrations around them.
    """
    st.markdown(html_7, unsafe_allow_html=True)
    im1 = Image.open('download.png')
    im2 = Image.open('download-1.png')
    col1, col2 = st.columns(2)
    with col1:
        st.image(im1, caption = "Top 10 heavily listed neighbourhoods")
    with col2:
        st.image(im2, caption = "listing density of the most listed areas")
        
    im3 = Image.open('download (14).png')
    st.image(im3, caption="Top 10 hosts by number of listings")
    st.markdown("It is evident that businesses operate on a quite larger scale than homeowners. We shall see more examples of this later", unsafe_allow_html=True)
    
    st.markdown("#")
    st.header("Prices and Neighbourhoods")
    im4 = Image.open('download (1).png')
    st.image(im4, caption="Top costliest neighbourhoods")
    st.markdown("Unsurprisingly, these are the main city centers in Chicago Metropolitan Area", unsafe_allow_html=True)
    
    im5 = Image.open('download (2).png')
    st.image(im5, caption="Daily price heatmap across the entire region")
    
    im6 = Image.open('newplot.jpg')
    st.image(im6, caption="Areas with their most popular room type")
    
    st.markdown("We should generate a clear idea with the following regarding the effect of various features on the daily prices. As well will see the correlations later, one of the main factor is the Room Type of stay", unsafe_allow_html=True)
    
    im7 = Image.open('download (3).png')
    st.image(im7, caption= "Price distribution across room types and populous regions")
    
    im8 = Image.open('download (4).png')
    st.image(im8, caption = "Price and listing features correlation {open image for viewing}", output_format='PNG')
    
    im9 = Image.open('download (5).png')
    st.image(im9, caption="pricing and popularity")
    
    html_8 = """
    <h3><b> We can conlude from this section that:</b> </h3>
    <ul>
         <li> <h3> Entire apartment Airbnb are the most costliest ones, while also being the most heavily listed </h3> </li>
         <li> <h3> Shared homes are cheaper, less frequently listed but feature in a couple of the densely listed neighbourhoods, which are major city centres </h3> </li>
         <li> <h3> More reviewed, hence popular listings are cheaper than others. Entire apartment listings are quite variable but it does stick to the aforementioned trend more often than not. You will only find entire apartment listings which are also decently popular on the higher end of costs. We can explore a price prediction on the basis of these features </h3> </li>
         <li> <h3> As the minimum days of stay increases, the cost decreases exponentially. </h3> </li>
    </ul>
    <br>
    """
    st.markdown(html_8, unsafe_allow_html=True)
    
    st.header("Descriptions and Names")
    st.markdown("Let's start by looking at the most common words/phrases that are listed in the name and description of the Airbnbs.")
    
    col3, col4 = st.columns(2)
    with col3:
        im10 = Image.open('download (15).png')
        st.image(im10, caption="Common words in listing names")
    with col4:
        im11 = Image.open('download (16).png')
        st.image(im11, caption="Common words in listing descriptions")
        
    st.markdown("Top n-grams of the 200 most popular listings")
    im12 = Image.open('download (6).png')
    st.image(im12)
    
    html_9 = """
    <ul>
         <li> <h3> Popular rooms are the ones with access to amenities readily, as well as location advantages. You can further analyse to confirm this by the price and amenities features corresponding to a listing in the dataset </h3> </li>
         <li> <h3> The immediate neighbourhood is heavily mentioned in the name, whereas the description constains the features of the listing, which is heavily correlated with the price, as we will see ahead </h3> </li>
    </ul>
    <br>
    """
    st.markdown(html_9, unsafe_allow_html=True)    






if box == "Feature Engineering":
    
    st.header("Feature data formatting and extraction")
    html_10 = """
    <div style="background-color:#FFE6A0 ;padding:10px">
    <p> <h3> In order to build a model for our use-case of price prediction, we need to correctly identify the key correlators in the data and format the input type to get
    the maximum positive feedback from the model. Thus we shall look at the creation and removal of existing features using statistical modelling and NLP techniques, in order to prime the dataset.</h3></p>
    <br>    
    """
    st.markdown(html_10, unsafe_allow_html=True)
    
    st.header("We wish to analyse the Amenities feature as it will be a key feature in prediction of prices of a listing")
    
    html_11 = """
    <p> <h3> We begin by lemmatizing the Amenities feature set. We create a bag of words model using the amenities we are interested in, for feature use in the dataset. 
    We will subsequently make a document-term matrix, and convert the sparse representation into a numpy array, which will be converted to append to our original dataframe. </h3></p>
    """
    st.markdown(html_11, unsafe_allow_html=True)
    
    im13 = Image.open('Capture1.jpg')
    st.image(im13, caption="Document Term matrix")
    st.markdown("<br>", unsafe_allow_html=True)
    
    im14 = Image.open('Capture2.jpg')
    st.image(im14, caption="Sparse representation dataframe")
    st.markdown("This sparse representation form will benefit us over one-hot encoding, as we wish to see each of the terms of the features weightage on the price prediction model", unsafe_allow_html=True)
    st.markdown("Apart from Amenities, the rest of the free-text features will be dropped,as well as the redundant ones after analysisng all multi-collinearities")
    
    im15 = Image.open('download (7).png')
    st.image(im15, caption="Collinearity heatmap after dropping most of the redundant and extreme correlating features")
    st.markdown("We shall go ahead with these features <br>", unsafe_allow_html=True)
    
    st.markdown("When it comes to the distribution of the values of the features, bar availability_365, rest of the features are very skewed. Therefore we can log transform these to a normal distribution. We will still have features not benefitted from this due to their irregular distribution and large number of 0 values. We will standardize all later on using scikit-learn <br>", unsafe_allow_html=True)
    
    im16 = Image.open('download (9).png')
    st.image(im16, caption="Correcting the skewed data distribution of the numerical features")
    
    
  
    
  
    
if box == "Model Building":
    
    st.header("Building and selecting appopriate prediction models for our use-case")
    html_12 = """
    <div style="background-color:#FFE6A0 ;padding:10px">
    <p> <h3> For this prediction problem , we shall look into building a strong fitting regression model using various novel techniques at hand. Beginning with a simple
    linear regresison as a baseline model, we will look into solving the same with Support Vector Machines, Ensemble Decision Tree based model and lastly a Neural Network.
    we will benchmark and evaluate all before settling on the most likely accurate predictor.</h3></p>
    <br>    
    """
    st.markdown(html_12, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<h3><b> 1. </b2></h3>", unsafe_allow_html=True)
    st.markdown("""The linear regresion model, as expected performs very poorly. Being a very biased interpolator, it does not fit well on our test dataset at all, and 
                  proceeds to give high error scores""", unsafe_allow_html=True)
                 
    im17 = Image.open('regression.jpg')
    st.image(im17, caption = "Linear regression scores")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<h3><b> 2. </b2></h3>", unsafe_allow_html=True)
    st.markdown("The SVM regressor behaves similary, albeit fitting on the test data set a bit better than the linear regressor. However it too ends up being a biased model to work with", unsafe_allow_html=True)
    im17b = Image.open('svm.jpg')
    st.image(im17b, caption="SVM regression score")
    st.markdown("#")
    
    st.markdown("<h3><b> 3. </b2></h3>", unsafe_allow_html=True)
    st.markdown("An ensemble model like vanilla XGBoost, built on the functionalities of a decision tree works the best in our use case. It has an accurate validation score, and does not overfit or underfit on the given data. Let us look at an untuned XGBoost model scores", unsafe_allow_html=True)
    im18 = Image.open('xgb_vanilla.jpg')
    st.image(im18, caption = "XGB base model scores")
    st.markdown("#")
    
    st.markdown("Using the feature weighatages, we select the top important features to tune the hyperparameters using Gridsearch CV with a 3 fold Cross Validation. We observe an increase in the accuracy in the dest set, as well as a more suitav=ble distribution of the predicted data with accordance to the test set", unsafe_allow_html=True)
    im19 = Image.open('xgb_tuned.jpg')
    st.image(im19, caption="Results after suitable tuning the hyperparameters")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    with col5:
        im20 = Image.open('download (10).png')
        st.image(im20, caption = "Prediction Plot")
    with col6:
        im21 = Image.open('download (11).png')
        st.image(im21, caption= "predicted and actual values distribution")
        
    st.markdown("#")
    st.markdown("The most weighted features after fitting on the tuned model are found out: ")
    
    im22 = Image.open('download (12).png')
    st.image(im22)
    
    st.markdown("#")
    
    html_13 = """
    <ul>
         <li> <h3> As expected, the room type being an entire place is one of the key decided factors for prices. Next up in importance is the accomodation size, which can be inferred directly from the fact how high it ranks in the user preferences while booking a room. This has been directly employed to optimize the selection process by the design decisions in the app/website. </h3> </li>
         <li> <h3> We find gym being a key factor too, although it's scarcity in data, coupled with the positive correlations we found in the sample we had, made way into a higher key feature. </h3> </li>
         <li> <h3> The key neighbourhoods featured in the list also rank high among the most listed neighbourhoods, as well as being costly and located in city centres. Their prices are also affected seasonally. </h3> </li>
         <li> <h3> After room details and neighbourhoods, another importance feature in determining the price is the number of listings by the same host. This can be accredited to the skew due to the monopoly of business listings, which manage a lot of properties. However, this very well may drive up the prices due to their reliability. </h3> </li>
    </ul>
    <br>
    """
    st.markdown(html_13, unsafe_allow_html=True)
    st.markdown("#")
    
    st.markdown("<h3><b> 4. </b2></h3>", unsafe_allow_html=True)
    st.markdown("We shall now observe the performance of a Neural Network with L1 regularization for the given task. We will use ReLU activation function, a linear function for the output layer and a mean sqaured loss as this is a regression task.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    im23 = Image.open('Capture-nn.jpg')
    st.image(im23, caption = "The structure of the Keras neural network built to fit our task")
    st.markdown("<br>", unsafe_allow_html=True)
    
    im24 = Image.open('Capture-nn-f.jpg')
    st.image(im24, caption="Loss and error at the completion of run epochs" )
    st.markdown("<br>", unsafe_allow_html=True)
    
    im25 = Image.open('download (13).png')
    st.image(im25, caption="Predicted values vs Actual values")
    st.markdown("<br>", unsafe_allow_html=True)
    
    html_14 = """
    <ul>
         <li> <h3> Even after priming the neural network with relevant activation fucntions and parameters, and the data batch having the most important features, we do not see an improvement over the XGBoost model, although it's close. </h3> </li>
         <li> <h3> Such an ensemble model is less prone to overfitting than a neural network model, while giving accurate prediction boundaries. </h3> </li>
         <li> <h3> So we would proceed with all prediction tasks using this tuned XGBoost model. It still doesn't explain the variation in the data, which may be due to important factors not present in the dataset. </h3> </li>
    </ul>
    <br>
    """
    st.markdown(html_14, unsafe_allow_html=True)
    
    
    



elif box == "Price Prediction":
    html_temp_3 = """
    <div style="background-color:#FFE6A0 ;padding:10px">
    <p> <h2> Get the price estimate of a listing based on user preferences </h2></p>
    <br>
    """
    st.markdown(html_temp_3, unsafe_allow_html=True)
    html_4 = """ <h3> Enter preferences </h3> """
    st.markdown(html_4, unsafe_allow_html=True)
    
    with st.form(key='user_dat'):
        room = st.selectbox("Room type", ["room_type_Entire Home/Apartment", "room_type_Private", "room_type_Shared", "room_type_Hotel"])
        area = st.selectbox("Select neighbourhood", ['neighbourhood_Near North Side', 'neighbourhood_Albany Park', 
       
       'neighbourhood_Loop', 
       'neighbourhood_Lincoln Park', 'neighbourhood_Washington Park',
        'neighbourhood_Lake View',
       'neighbourhood_Logan Square', 'neighbourhood_Near West Side',
       
       'neighbourhood_East Garfield Park',
       'neighbourhood_South Chicago', 'neighbourhood_Uptown',
       'neighbourhood_Belmont Cragin', 'neighbourhood_Oakland',
       'neighbourhood_Englewood', 'neighbourhood_New City',
       'neighbourhood_Austin', 'neighbourhood_Dunning', 'neighbourhood_Lincoln Highlands'])
        acc = st.text_input("Number of people to acommodate")
        bedroom = st.text_input("Number of bedrooms")
        nights = st.text_input("Number of nights stay")
        amen = st.multiselect("Amenities required", ["gym",	"dishwasher", "kitchen", "indoor_fireplace", "iron", 
                                                     "washer", "heating", "hair_dryer", "stove", "hot_water", "microwave", "dryer"])
        
        submit = st.form_submit_button("Predict")
        
        if submit:
            X['longitude'] = -122.238732	
            X['latitude'] = 37.809807	
            X['reviews_per_month'] = -0.6654210618502802
            X['calculated_host_listings_count'] = 1.2938878937753027
            X['availability_365'] = 191.607054
            X['review_scores_rating'] = 0.465795193102471
            X['number_of_reviews'] = 1.6500842530785723
            X['maximum_nights'] = 5.6961927253466635       
                                
            X['accommodates'] = math.log(int(float(acc)))
            X['bedrooms'] = math.log(int(float(bedroom)))
            X['minimum_nights'] = math.log(int(float(nights)))
            
            input_room(X, roomtype= room)
            input_area(X, areatype= area)
            input_amen(X, amenities= amen)
            
            a = X.columns
            st.text(a)
            
            if int(float(acc)) > 16 and int(float(acc)) <= 0:
                st.error("Number of accomodation must be between 1 to 16 people")
                
            if int(float(bedroom)) > 12 and int(float(bedroom)) < 1:
                st.error("Number of beds must not exceed 12 and fall below 1")
            
            if int(float(nights)) <= 0:
                st.error("Minimum 1 night stay required")
                
            val = predict_xgb(X, filename= 'model_xgb.json')
            st.text("Predicted price in Dollars per Night")
            st.info(val)