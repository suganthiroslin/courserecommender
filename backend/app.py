from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os

from flask_cors import CORS

# ============================================================
# NLP IMPORTS
# ============================================================

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Allow requests from Netlify frontend
CORS(app)


# ============================================================
# 2. DATABASE CONFIGURATION
# ============================================================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///courses.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ============================================================
# 3. COURSE DATABASE MODEL
# ============================================================

class Course(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    course_title = db.Column(
        db.String(500),
        nullable=False
    )


# ============================================================
# 4. LOAD DATASET INTO DATABASE
# ============================================================

def load_courses():

    # --------------------------------------------------------
    # Find the folder where app.py is located
    # --------------------------------------------------------

    base_folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    # --------------------------------------------------------
    # Dataset path
    # --------------------------------------------------------

    csv_path = os.path.join(
        base_folder,
        "coursea_data.csv"
    )

    # --------------------------------------------------------
    # Check whether dataset exists
    # --------------------------------------------------------

    if not os.path.exists(csv_path):

        print()
        print("ERROR: Dataset file not found!")
        print()

        print("Python is looking for:")
        print(csv_path)

        print()

        print("Make sure your dataset is named:")
        print("coursea_data.csv")

        print()

        return

    # --------------------------------------------------------
    # Check whether database already has courses
    # --------------------------------------------------------

    if Course.query.count() > 0:

        print(
            "Courses already loaded into database."
        )

        return

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    print()
    print("Loading dataset...")
    print()

    data = pd.read_csv(
        csv_path
    )

    # --------------------------------------------------------
    # Display columns
    # --------------------------------------------------------

    print(
        "Columns found in dataset:"
    )

    print(
        data.columns.tolist()
    )

    print()

    # --------------------------------------------------------
    # Check course_title column
    # --------------------------------------------------------

    if "course_title" not in data.columns:

        print(
            "ERROR: 'course_title' column not found."
        )

        print()

        print(
            "Available columns are:"
        )

        print(
            data.columns.tolist()
        )

        return

    # --------------------------------------------------------
    # Remove empty course titles
    # --------------------------------------------------------

    data = data.dropna(
        subset=["course_title"]
    )

    # --------------------------------------------------------
    # Remove duplicate courses
    # --------------------------------------------------------

    data = data.drop_duplicates(
        subset=["course_title"]
    )

    # --------------------------------------------------------
    # Insert courses into SQLite
    # --------------------------------------------------------

    for title in data["course_title"]:

        course = Course(
            course_title=str(title).strip()
        )

        db.session.add(course)

    db.session.commit()

    print(
        "Dataset successfully loaded!"
    )

    print(
        "Total courses:",
        Course.query.count()
    )

    print()


# ============================================================
# 5. NLP PREPROCESSING
# ============================================================

def preprocess_text(text):

    # --------------------------------------------------------
    # Convert text to lowercase
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # Remove special characters and punctuation
    # Keep only letters and numbers
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove extra spaces
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    words = text.split()

    # --------------------------------------------------------
    # Remove English stop words
    # --------------------------------------------------------

    words = [

        word

        for word in words

        if word not in ENGLISH_STOP_WORDS

    ]

    # --------------------------------------------------------
    # Join cleaned words
    # --------------------------------------------------------

    cleaned_text = " ".join(words)

    return cleaned_text


# ============================================================
# 6. RECOMMENDATION FUNCTION
# ============================================================

def recommend_courses(user_input):

    # --------------------------------------------------------
    # Get courses from database
    # --------------------------------------------------------

    courses = Course.query.all()

    if not courses:

        return []

    # --------------------------------------------------------
    # Get course titles
    # --------------------------------------------------------

    course_titles = [

        course.course_title

        for course in courses

    ]

    # ========================================================
    # NLP PREPROCESSING
    # ========================================================

    # --------------------------------------------------------
    # Clean all course titles
    # --------------------------------------------------------

    cleaned_course_titles = [

        preprocess_text(title)

        for title in course_titles

    ]

    # --------------------------------------------------------
    # Clean user input
    # --------------------------------------------------------

    cleaned_user_input = preprocess_text(
        user_input
    )

    # --------------------------------------------------------
    # Display NLP processing information
    # This helps during testing/debugging
    # --------------------------------------------------------

    print()
    print("Original User Input:")
    print(user_input)

    print()

    print("After NLP Preprocessing:")
    print(cleaned_user_input)

    print()

    # --------------------------------------------------------
    # Check if preprocessing removed everything
    # --------------------------------------------------------

    if not cleaned_user_input:

        return []

    # ========================================================
    # TF-IDF
    # ========================================================

    # --------------------------------------------------------
    # TF-IDF Vectorizer
    #
    # Stop words are already removed above using NLP.
    # Therefore stop_words=None is used here.
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words=None
    )

    # --------------------------------------------------------
    # Convert course titles into TF-IDF vectors
    # --------------------------------------------------------

    course_vectors = vectorizer.fit_transform(
        cleaned_course_titles
    )

    # --------------------------------------------------------
    # Convert cleaned user input into TF-IDF vector
    # --------------------------------------------------------

    user_vector = vectorizer.transform(
        [cleaned_user_input]
    )

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    # --------------------------------------------------------
    # Calculate similarity between user query
    # and every course
    # --------------------------------------------------------

    similarity_scores = cosine_similarity(
        user_vector,
        course_vectors
    )[0]

    # ========================================================
    # PREDICTION / RANKING
    # ========================================================

    # --------------------------------------------------------
    # Create recommendation list
    # --------------------------------------------------------

    recommendations = []

    for index, course in enumerate(courses):

        recommendations.append({

            "id": course.id,

            "course_title": course.course_title,

            "score": round(
                float(similarity_scores[index]) * 100,
                2
            )

        })

    # --------------------------------------------------------
    # Sort recommendations
    # Highest similarity first
    # --------------------------------------------------------

    recommendations.sort(

        key=lambda x: x["score"],

        reverse=True

    )

    # --------------------------------------------------------
    # Return top 5 predictions
    # --------------------------------------------------------

    return recommendations[:5]


# ============================================================
# 7. HOME PAGE
# ============================================================

@app.route("/")
def home():

    return "Course Recommendation API is running!"


# ============================================================
# 8. RECOMMENDATION API
# ============================================================

@app.route(
    "/api/recommend",
    methods=["POST"]
)
def recommendation_api():

    # --------------------------------------------------------
    # Get JSON request
    # --------------------------------------------------------

    data = request.get_json()

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if not data:

        return jsonify({

            "error":
            "No input data provided."

        }), 400

    # --------------------------------------------------------
    # Get user query
    # --------------------------------------------------------

    user_input = data.get(
        "query",
        ""
    ).strip()

    # --------------------------------------------------------
    # Check empty query
    # --------------------------------------------------------

    if not user_input:

        return jsonify({

            "error":
            "Please enter your skills or interests."

        }), 400

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendations = recommend_courses(
        user_input
    )

    # --------------------------------------------------------
    # Check whether recommendations exist
    # --------------------------------------------------------

    if not recommendations:

        return jsonify({

            "error":
            "No matching courses found."

        }), 404

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return jsonify({

        "query": user_input,

        "recommendations":
            recommendations

    })


# ============================================================
# 9. DATABASE INITIALIZATION
# ============================================================

# IMPORTANT:
# This must be OUTSIDE the __main__ block.
#
# Render uses:
#
#     gunicorn app:app
#
# Gunicorn imports this file instead of running
# the __main__ section.
#
# Therefore database creation and CSV loading
# must happen when the application is imported.
# ============================================================

with app.app_context():

    db.create_all()

    load_courses()


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("   COURSE RECOMMENDATION SYSTEM")
    print("======================================")
    print()

    print(
        "Starting Flask server..."
    )

    print()

    app.run(
        debug=True
    )
