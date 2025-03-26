# Importing necessary Django modules for handling database operations and rendering templates
from django.db.models import Count, Q  # Used for database queries and aggregation
from django.shortcuts import render, redirect, get_object_or_404  # Used for rendering pages, redirecting, and retrieving objects

# Importing standard libraries
import datetime  # Handling date and time operations
import openpyxl  # Reading and writing Excel files
import string  # Working with string manipulations
import re  # Regular expressions for text processing

# Importing data analysis and visualization libraries
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # Statistical data visualization

# Importing machine learning libraries
from sklearn.ensemble import VotingClassifier  # Combining multiple models for better accuracy
from sklearn.feature_extraction.text import CountVectorizer  # Converting text data into numerical representation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score  # Performance evaluation metrics
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn import svm  # Support Vector Machine classifier
from sklearn.model_selection import train_test_split  # Splitting dataset into training and testing
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier

# Suppressing warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Importing database models from the Remote_User app
from Remote_User.models import ClientRegister_Model, profile_identification_type, detection_ratio, detection_accuracy

# ----------------------------- User Authentication (Login) -----------------------------

def login(request):
    if request.method == "POST" and 'submit1' in request.POST:  # Checking if the request is a POST request (form submission)
        username = request.POST.get('username')  # Retrieving username from form input
        password = request.POST.get('password')  # Retrieving password from form input

        try:
            # Checking if the entered username and password match a record in the database
            enter = ClientRegister_Model.objects.get(username=username, password=password)
            request.session["userid"] = enter.id  # Storing user ID in session for maintaining login state
            return redirect('ViewYourProfile')  # Redirecting the user to their profile page upon successful login
        except:
            pass  # If login fails, do nothing and reload the login page

    return render(request, 'RUser/login.html')  # Rendering the login page

# ----------------------------- User Registration -----------------------------

def Register1(request):
    if request.method == "POST":  # Checking if the request is a POST request (form submission)
        # Retrieving user registration details from the form
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')

        # Saving the new user details in the database
        ClientRegister_Model.objects.create(
            username=username, email=email, password=password, phoneno=phoneno,
            country=country, state=state, city=city
        )

        return render(request, 'RUser/Register1.html')  # Reloading the registration page after successful registration
    else:
        return render(request, 'RUser/Register1.html')  # Rendering the registration page if method is not POST

# ----------------------------- Viewing User Profile -----------------------------

def ViewYourProfile(request):
    userid = request.session['userid']  # Retrieving logged-in user ID from session
    obj = ClientRegister_Model.objects.get(id=userid)  # Fetching user details from the database
    return render(request, 'RUser/ViewYourProfile.html', {'object': obj})  # Rendering the profile page with user details

# ----------------------------- Fake Profile Detection -----------------------------

def Predict_Profile_Identification_Status(request):
    if request.method == "POST":  # Checking if the request is a POST request (form submission)
        # Retrieving input data from the form
        prof_idno = request.POST.get('prof_idno')
        name = request.POST.get('name')
        screen_name = request.POST.get('screen_name')
        statuses_count = request.POST.get('statuses_count')
        followers_count = request.POST.get('followers_count')
        friends_count = request.POST.get('friends_count')
        created_at = request.POST.get('created_at')
        location = request.POST.get('location')
        default_profile = request.POST.get('default_profile')
        prf_image_url = request.POST.get('prf_image_url')
        prf_banner_url = request.POST.get('prf_banner_url')
        prf_bgimg_https = request.POST.get('prf_bgimg_https')
        prf_text_color = request.POST.get('prf_text_color')
        profile_image_url_https = request.POST.get('profile_image_url_https')
        prf_bg_title = request.POST.get('prf_bg_title')
        profile_background_image_url = request.POST.get('profile_background_image_url')
        description = request.POST.get('description')
        Prf_updated = request.POST.get('Prf_updated')

        # Loading dataset for training the model
        df = pd.read_csv('Profile_Datasets.csv')

        # Function to clean text data
        def clean_text(text):
            text = text.lower()  # Convert text to lowercase
            text = re.sub('\[.*?\]', '', text)  # Remove text in square brackets
            text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove links
            text = re.sub('<.*?>+', '', text)  # Remove HTML tags
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
            text = re.sub('\n', '', text)  # Remove newlines
            text = re.sub('\w*\d\w*', '', text)  # Remove words containing numbers
            return text

        # Cleaning name column in dataset
        df['processed_content'] = df['name'].apply(lambda x: clean_text(x))

        # Mapping labels to binary values (0 for fake, 1 for genuine)
        def apply_results(label):
            return 0 if label == 0 else 1

        df['results'] = df['Label'].apply(apply_results)

        # Convert text into numerical representation
        cv = CountVectorizer(lowercase=False)
        X = cv.fit_transform(df["id"].apply(str))  # Transforming IDs into numerical format
        y = df['results']  # Target labels

        # Splitting dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Training SVM model
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        models = [('svm', lin_clf)]  # Adding SVM model to voting classifier

        # Training K-Nearest Neighbors classifier
        kn = KNeighborsClassifier()
        kn.fit(X_train, y_train)
        models.append(('KNeighborsClassifier', kn))  # Adding KNN model to voting classifier

        # Combining models using Voting Classifier
        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)

        # Predicting profile authenticity
        vector1 = cv.transform([prof_idno]).toarray()  # Transforming input into numerical format
        predict_text = classifier.predict(vector1)

        # Converting prediction to human-readable format
        prediction = int(str(predict_text).replace("[", "").replace("]", ""))
        val = 'Fake Profile' if prediction == 0 else 'Genuine Profile'

        # Saving results in the database
        profile_identification_type.objects.create(
            prof_idno=prof_idno, name=name, screen_name=screen_name, statuses_count=statuses_count,
            followers_count=followers_count, friends_count=friends_count, created_at=created_at,
            location=location, description=description, Prf_updated=Prf_updated, Prediction=val
        )

        return render(request, 'RUser/Predict_Profile_Identification_Status.html', {'objs': val})  # Displaying result

    return render(request, 'RUser/Predict_Profile_Identification_Status.html')  # Rendering page if method is not POST
