# Spotify Songs ML Project

## Overview
The Spotify Songs ML Project aims to predict the popularity of Spotify tracks using audio features, release information, and playlist context. The project leverages a dataset sourced from Kaggle ([30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)), which contains approximately 32,833 songs with over 23 features—including track details, audio features (e.g., danceability, energy, loudness, tempo), temporal data (release date), and playlist information.

## Research Question
**Can we predict a Spotify song's popularity using its audio characteristics, release details, and associated playlist/genre information?**

## Project Goal
Develop a robust regression model to estimate the popularity score of Spotify songs (ranging from 0 to 100) and identify the key factors that influence track popularity.


## Methodology

### Data Preprocessing
1. **Data Loading:**  
   Loaded the dataset from Kaggle, which includes ~32,833 songs with unique identifiers, descriptive fields, audio features, temporal data, and playlist information. The target variable, `track_popularity`, is numeric and ranges from 0 to 100.

2. **Date Conversion & Feature Extraction:**  
   - Converted `track_album_release_date` to a datetime type using `pd.to_datetime(errors='coerce')`.
   - Extracted `release_year` and `release_month`; computed `song_age` (2023 – release_year) and derived a `decade` column.

3. **Text Normalization & Type Conversion:**  
   - Standardized text fields (e.g., `track_name`, `track_artist`, `track_album_name`) by stripping extra whitespace and converting text to lowercase.
   - Ensured uniform data types by converting all object-type columns to the Pandas "string" dtype.

4. **Duplicate Handling & Filtering:**  
   - Aggregated duplicates by `track_id`, song attributes, and tempo rounding.
   - Filtered out songs with a tempo below 20 BPM or a duration less than 30 seconds.

### Exploratory Data Analysis (EDA)
- **Descriptive Statistics:**  
  Examined feature distributions and outliers using histograms, box plots, and scatter plots.
- **Correlation Analysis:**  
  Computed a correlation matrix to identify relationships among features (e.g., a strong positive correlation between loudness and energy).
- **Temporal & Categorical Analysis:**  
  Analyzed trends over time (via the `decade` column) and the distribution of playlist genres through count plots and pie charts.
- **Textual Analysis:**  
  Generated a word cloud of artist names to visually highlight frequent artists in the dataset.

### Feature Selection
- **Approach:**  
  Employed a multi-model strategy using Lasso, Ridge, LinearSVR, Gradient Boosting Regressor, and Random Forest Regressor.
- **Criteria:**  
  A feature was retained if selected (non-zero coefficient or importance) by at least 4 out of the 5 models.
- **Final Feature Set (with Index):**
  
  1. **danceability**  
  2. **energy**  
  3. **key**  
  4. **loudness**  
  5. **mode**  
  6. **speechiness**  
  7. **acousticness**  
  8. **instrumentalness**  
  9. **liveness**  
  10. **valence**  
  11. **tempo**  
  12. **duration_ms**  
  13. **track_album_release_year**  
  14. **track_album_release_month**  
  15. **song_age**  
  16. **decade**  
  17. **genre_edm**  
  18. **genre_latin**  
  19. **genre_pop**  
  20. **genre_r&b**  
  21. **genre_rap**  
  22. **genre_rock**  
  23. **track_popularity** *(target variable)*

### Model Selection & Evaluation
- **Data Split:**  
  The dataset was split into training (80%) and test (20%) sets.
- **Models Evaluated:**  
  We evaluated several regression models:
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - AdaBoost Regressor  
  - Gradient Boosting Regressor (GBM)  
  - Support Vector Regressor (SVR)  
  - XGBoost Regressor
- **Evaluation Metrics:**  
  Models were assessed using:
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - Mean Absolute Error (MAE)  
  - R² Score  
  - Root Mean Squared Logarithmic Error (RMSLE)
- **Visual Analysis:**  
  Scatter plots comparing predicted vs. actual values were used to visually evaluate model performance.
- **Final Summary:**  
  Model metrics were aggregated into a summary table, allowing for easy comparison and identification of the best-performing model.

## Results and Future Work
- **Key Findings:**  
  Features such as energy, danceability, loudness, and genre indicators are significant predictors of track popularity.


