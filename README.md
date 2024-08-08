# Recipe Recommender System

## Overview

This project is a recipe recommender system that uses the concepts of TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity. The system takes a list of ingredients as input and recommends an appropriate recipe that you can cook with those ingredients.

## Features

- **Ingredient Input**: Users can input a list of ingredients they have.
- **Recipe Recommendation**: The system recommends the most relevant recipe based on the provided ingredients.
- **Similarity Calculation**: Uses TF-IDF and cosine similarity to determine the relevance of recipes.

## Requirements

- Python 3.6 or higher
- pandas
- scikit-learn
- nltk

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Recipe-Recommender-System.git
    cd Recipe-Recommender-System
    ```

2. Install the required packages:
    ```bash
    pip install pandas scikit-learn nltk
    ```

3. Download NLTK stopwords:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Usage

1. Prepare your dataset of recipes and ingredients. Ensure that your DataFrame has a column named 'Ingredients'.

2. Create a list of custom common words if needed:
    ```python
    common_words = ['cup', 'teaspoon', 'tablespoon', 'salt', 'taste', 'thinly', 'gram', 'chop', 'fine']
    ```

3. Remove common words from the 'Ingredients' column:
    ```python
    df['Ingredients'] = df['Ingredients'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in set(common_words)]))
    ```

4. Convert any list entries in the 'Ingredients' column to strings:
    ```python
    df['Ingredients'] = df['Ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    ```

5. Implement TF-IDF and cosine similarity to recommend recipes:
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Example ingredients input
    input_ingredients = ["tomato", "chicken", "rice"]

    # Convert input ingredients to a single string
    input_ingredients_str = ' '.join(input_ingredients)

    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the ingredients data
    tfidf_matrix = vectorizer.fit_transform(df['Ingredients'])

    # Transform the input ingredients
    input_tfidf = vectorizer.transform([input_ingredients_str])

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()

    # Get the index of the most similar recipe
    most_similar_recipe_index = cosine_similarities.argmax()

    # Get the recommended recipe
    recommended_recipe = df.iloc[most_similar_recipe_index]

    print("Recommended Recipe:", recommended_recipe)
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project uses data and algorithms from the scikit-learn and nltk libraries.
- Special thanks to all contributors and supporters.

