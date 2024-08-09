import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from streamlit_option_menu import option_menu
import pickle
from sklearn.metrics.pairwise import cosine_similarity


with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Recipe Recommender System","Data Analysis","Recommender"],
    )


if selected == "Recipe Recommender System":
    st.markdown(" # Recipe Recommender System")

    st.header('ABOUT')
    st.subheader("Introduction")
    st.write("Our system helps you find the best recipes based on the ingredients you have on hand. Whether you're trying to make the most of what's in your pantry or looking for new culinary inspirations, our recommender has you covered.")

    st.subheader("How It Works")
    st.write("Using advanced text analysis techniques like TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity, our system matches your ingredients with a vast database of recipes to find the best matches. Simply enter your ingredients, and let our algorithm do the rest!")
    
    st.subheader("Dataset In Use :-")
    df = pd.read_csv("IndianFoodDatasetCSV.csv")
    st.dataframe(df,height=500)
    
    st.subheader("Features Of Our Model")
    st.write("- **Ingredient-Based Recommendations**: Enter any combination of ingredients and receive tailored recipe suggestions.")
    st.write("- **Detailed Recipe Information**: Get a comprehensive list of ingredients, step-by-step cooking instructions, and preparation time for each recipe.")
    st.write("- **Save and Share**: Save your favorite recipes and share them with friends and family on social media.")

    st.image("cmei.jpg")
    
    st.subheader("Benefits")
    st.write("- **Reduce Food Waste**: Make the most of your available ingredients.")
    st.write("- **Discover New Recipes**: Explore a variety of recipes and find new favorites.")
    st.write("- **Save Time**: Get quick and accurate recipe recommendations without the need for extensive searches.")
    st.write("- **Customize Your Experience**: Tailor your search based on dietary needs and preferences.")
    
    st.subheader("Technical Details")
    st.write("Our system is built using Python, with key libraries including scikit-learn for machine learning and Streamlit for the web interface. The recommendation engine leverages TF-IDF for text analysis and cosine similarity for matching recipes.")
    
    st.subheader("User Guidance")
    st.write("To get started, simply enter the ingredients you have in the search box and click Get Recommendations and explore detailed information for each recipe.")
    st.write("You can enter one or more ingredients seperated by a comma.")
    st.write("More the ingredients you give to the recommender the better recommendation it gives.")
    st.write("If the dataset does not have recipes for the ingredients you gave to the recommender it returns the default first five recipes")

if selected == "Data Analysis":
    df = pd.read_csv("IndianFoodDatasetCSV.csv")
    st.markdown('''
                # Learn About The Dataset
                ''')
    st.subheader("what is the range of our cooking time")

    st.write("**Most Common Cooking Time Ranges:** The most common cooking time range is 45-60 minutes, followed closely by 15-30 minutes and 30-44 minutes. These three ranges dominate, suggesting that most recipes in the dataset can be prepared within an hour.")
    st.write("**Shorter Cooking Times:** There is a significant number of recipes that take 0-15 minutes, indicating a good selection of quick recipes.")
    st.write("**Moderate Cooking Times:** The 60-120 minutes range is also well represented, showing that there are many recipes that require around 1 to 2 hours of cooking time.")
    st.write("**Long Cooking Times**: Recipes that take between 120-240 minutes (2 to 4 hours) and 240 minutes to 1 day are less common, indicating that fewer recipes require long cooking times. There are very few recipes that take 1 day or more.")

    df_tt=pd.cut(x=df['TotalTimeInMins'],bins=[0,15,30,40,60,120,240,1440,3000], labels=['0-15','15-30','30-44','45-60','60-120','120-240','240-1 day','1 day+'])

    value_counts = df_tt.value_counts()

    fig, ax = plt.subplots()

    value_counts.plot(kind='bar', color='midnightblue', ax=ax)


    ax.set_facecolor('lightcyan')
    fig.patch.set_facecolor('lightcyan')
    ax.set_xlabel('Total Time in Mins')
    ax.set_ylabel('Count')
    ax.set_title('Total Time in Mins')

    st.pyplot(fig)


    st.subheader('Prep time vs cook time')
    st.write("**Median Times:** Preparation time is generally shorter than cooking time, with median prep time around 10 minutes and median cook time around 20 minutes.")
    st.write("**Variability:** Prep times are more consistent (less variable) compared to cook times, which show a broader range.")
    st.write("**Whiskers and Ranges:** Most preparation times fall between 5 and 35 minutes and most cooking times fall between 5 and 55 minutes.")
    st.write("**Outliers:** There are more outliers in prep time, indicating some recipes take unusually long to prepare. Cooking time outliers are fewer but indicate some recipes take significantly longer than the typical range.")


    prep_time_95th = df['PrepTimeInMins'].quantile(0.95)
    cook_time_95th = df['CookTimeInMins'].quantile(0.95)

    filtered_df = df[(df['PrepTimeInMins'] <= prep_time_95th) & (df['CookTimeInMins'] <= cook_time_95th)]

    fig, ax = plt.subplots(figsize=(10, 6))

    
    x = ax.boxplot([filtered_df['PrepTimeInMins'], filtered_df['CookTimeInMins']], patch_artist=True, vert=True)
    colors = ['blue', 'midnightblue']
    ax.set_facecolor('lightcyan')
    fig.patch.set_facecolor('lightcyan')
    for patch, color in zip(x['boxes'], colors):
        patch.set_facecolor(color)

    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Prep Time In Mins', 'Cook Time In Mins'])
    ax.set_ylabel('Time in Mins')
    ax.set_title('Zoomed-in Boxplot (Excluding Extreme Outliers)')
    st.pyplot(fig)

    st.subheader('Different Courses')

    st.write("The pie chart shows that lunch recipes are the most prevalent, accounting for 25.7% of the dataset, followed by a significant Otherscategory, indicating a variety of courses that do not fit into the traditional meal categories. It suggests that the dataset includes a wide range of recipes that might cater to unique or specific meal types not covered by other categories. category at 16.9%. Side dishes (14.4%), snacks (12.7%), and dinner (11.4%) also have substantial representations. Desserts (9.6%) and appetizers (9.3%) are less common but still present. This distribution indicates a comprehensive dataset with a variety of meal types, catering to diverse culinary needs and preferences, from main meals to snacks and desserts. The diversity ensures users have plenty of options for different meal times and dietary needs.")

    course_count = df['Course'].value_counts()
    threshold = 0.05 * course_count.sum()

    course_counts_mod = course_count[course_count >= threshold]
    course_counts_mod['Others'] = course_count[course_count < threshold].sum()

    
    fig, ax = plt.subplots()
    ax.set_facecolor('lightcyan')
    fig.patch.set_facecolor('lightcyan')
    colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43']
    course_counts_mod.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors, explode=[0.04 if i < len(course_counts_mod) - 1 else 0 for i in range(len(course_counts_mod))], ax=ax)

    ax.set_ylabel('') 
    ax.set_title('Course Distribution')

    st.pyplot(fig)

    st.subheader("Ingredient Count vs Prep Time")

    st.write("Recipes with a shorter prep time (under 100 minutes) show a wide range in the number of ingredients, from as few as 5 to over 40.")
    st.write("For longer prep times, the variation in the number of ingredients is less, with a general concentration around 10 to 20 ingredients.")
    st.write("There are a few outliers with exceptionally high prep times (close to 500 minutes), but these do not necessarily have a large number of ingredients.")

    df['IngredientsNum'] = df['Ingredients'].apply(lambda x: len(str(x).split(',')))

    lower_bound=df['PrepTimeInMins'].quantile(0.01)
    upper_bound=df['PrepTimeInMins'].quantile(0.99)
    df['PrepTimeInMins'] = np.clip(df['PrepTimeInMins'], lower_bound, upper_bound)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.scatter(df['PrepTimeInMins'],df['IngredientsNum'],color='midnightblue',marker='o')
    ax1.set_xlabel('Prep Time in Mins')
    ax1.set_ylabel('Number of Ingredients')
    ax1.set_title('Ingredients Count vs Prep Time')
    ax.set_facecolor('lightcyan')
    fig.patch.set_facecolor('lightcyan')
    st.pyplot(fig)


    st.subheader("Types of Diet")
    st.write("**Dominance of Vegetarian Diets:** The majority of the recipes (68.6%) are categorized as Vegetarian. This suggests that the dataset is heavily skewed towards vegetarian dishe")
    st.write("**Other Dietary Types:**")
    st.write("High Protein Vegetarian dishes make up 10.3% of the recipes, indicating a significant interest in protein-rich vegetarian meals.")
    st.write("Non-Vegetarian dishes account for 6.2%, showing a smaller, but still notable, representation of recipes including meat.")
    st.write("Eggetarian diets, which include eggs but exclude other meats, represent 5.0% of the dataset.")
    st.write("Others make up 9.9% of the dataset, likely including various specialized or mixed diets not covered by the main categories.")

    dietPlot = df['Diet']
    value_counts = dietPlot.value_counts()
    threshold = 0.05 * value_counts.sum()
    dietPlan_mod = value_counts[value_counts >= threshold]
    dietPlan_mod['others'] = value_counts[value_counts < threshold].sum()


    fig, ax = plt.subplots()
    colors2 = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087']
    dietPlan_mod.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors2, ax=ax)
    ax.set_facecolor('lightcyan')
    ax.set_title('Types of diet')
    fig.patch.set_facecolor('lightcyan')
    st.pyplot(fig)

    st.header("Key Ingredients")

    st.write("**Red Chilli**: This appears prominently, indicating that it's a common ingredient in the recipes, likely due to its usage in many spicy dishes.")

    st.write("**Sunflower**: This could refer to sunflower oil, a popular cooking oil, which is commonly used across various recipes.")

    st.write("**Turmeric Powder**: A staple in many dishes, especially in South Asian cuisines, it’s widely used for its flavor and color.")

    st.write("**Coriander and Dhania**: These refer to the same ingredient (coriander), which is common in many recipes.")

    st.write("**Teaspoon, Chopped, Finely**: These terms are frequent because they describe how ingredients are prepared, not specific ingredients themselves.")

    st.write("Common terms like **Teaspoon**, **Tablespoon**, **Cup**, and **Chopped** appear frequently, indicating standard units of measure and preparation methods.")

    # word cloud
    text_ingd = ' '.join(df['Ingredients'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='lightcyan').generate(text_ingd)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Standardized Ingredients')
    ax.set_facecolor('lightcyan')
    fig.patch.set_facecolor('lightcyan')
    st.pyplot(fig)

    st.subheader("Frequency of most common words")
    text = """
    **Teaspoon**: The most frequently occurring word, indicating that many recipes require precise measurements, and "teaspoon" is a common unit.

    **Cup**: Another common measurement unit, often used for liquids or bulk ingredients.

    **Tablespoon**: Like "teaspoon," this is frequently used, especially for slightly larger quantities of ingredients.

    **Finely**: These words suggest that many recipes require ingredients to be cut or prepared in a specific way, reflecting the importance of preparation techniques in cooking.
    """

    # Display the text using the write function
    st.write(text)

    text_freq = ' '.join(df['Ingredients'].astype(str))
    words = text_freq.split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(6)
    highest_freq_word = common_words[0]  # The first element has the highest frequency

# Remove hyphens from the highest frequency word
    highest_freq_word_no_hyphens = (highest_freq_word[0].replace('-', ''), highest_freq_word[1])

# Remove the highest frequency word from the common words list
    common_words_no_highest = [item for item in common_words if item != highest_freq_word]
    df_common_words = pd.DataFrame(common_words_no_highest, columns=['Words', 'Frequency'])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    df_common_words.plot(kind='barh', x='Words', y='Frequency', color='midnightblue', ax=ax)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Common Words')
    ax.set_title('Top 5 most common words in the dataset')
    ax.set_facecolor('lightcyan')
    fig.patch.set_facecolor('lightcyan')
    st.pyplot(fig)


    st.write("**To use the recommender system use the recommender button on the navigation bar for further details go to user guidance**")



    
if selected == "Recommender":
    with open('recipe_recommender_model.sav', 'rb') as file:
        tfv_matrix_diet = pickle.load(file)

    with open('recipe_fit.sav', 'rb') as file:
        tfv = pickle.load(file)

    df = pd.read_csv("cleaned_data.csv")

    def give_recipe_Rec(Ingredients):
        input_vector = tfv.transform([Ingredients])
        similarity_scores = cosine_similarity(input_vector, tfv_matrix_diet)
        top_indices = similarity_scores.argsort()[0][-5:]

        top_indices = [i for i in reversed(top_indices) if i < len(df)]
        recommendations = [
            {
                'RecipeName': df.iloc[i]['RecipeName'],
                'Ingredients': df.iloc[i]['Ingredients'],
                'Instructions': df.iloc[i]['Instructions'],
                'TotalTimeInMins': df.iloc[i]['TotalTimeInMins'],
                'LinkToRecipe': df.iloc[i]['URL']
            } for i in top_indices
        ]
        
        return recommendations

    st.title("Recipe Recommender")

    ingredients = st.text_input("Enter Ingredients")

    # Button to get recommendations
    if st.button("Get Recommendations"):
        if ingredients:
            recommendations = give_recipe_Rec(ingredients)
            for recipe in recommendations:
                st.subheader(f"Recipe: {recipe['RecipeName']}")
                st.write(f"**Ingredients**: {recipe['Ingredients']}")
                st.write(f"**Instructions**: {recipe['Instructions']}")
                st.write(f"**Preparation Time**: {recipe['TotalTimeInMins']} minutes")
                st.write(f"**Link To Recipe** : {recipe['LinkToRecipe']}")
    
       
