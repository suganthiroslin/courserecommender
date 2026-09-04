from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os


# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


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
# 5. RECOMMENDATION FUNCTION
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


    # --------------------------------------------------------
    # TF-IDF Vectorizer
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )


    # --------------------------------------------------------
    # Convert course titles into vectors
    # --------------------------------------------------------

    course_vectors = vectorizer.fit_transform(
        course_titles
    )


    # --------------------------------------------------------
    # Convert user input into vector
    # --------------------------------------------------------

    user_vector = vectorizer.transform(
        [user_input]
    )


    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarity_scores = cosine_similarity(
        user_vector,
        course_vectors
    )[0]


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
    # --------------------------------------------------------

    recommendations.sort(

        key=lambda x: x["score"],

        reverse=True

    )


    # --------------------------------------------------------
    # Return top 5
    # --------------------------------------------------------

    return recommendations[:5]


# ============================================================
# 6. HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# 7. RECOMMENDATION API
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
    # Return response
    # --------------------------------------------------------

    return jsonify({

        "query": user_input,

        "recommendations":
            recommendations

    })


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("   COURSE RECOMMENDATION SYSTEM")
    print("======================================")
    print()


    # --------------------------------------------------------
    # Create database
    # --------------------------------------------------------

    with app.app_context():

        db.create_all()

        load_courses()


    # --------------------------------------------------------
    # Start Flask
    # --------------------------------------------------------

    print(
        "Starting Flask server..."
    )

    print()

    app.run(
        debug=True
    )