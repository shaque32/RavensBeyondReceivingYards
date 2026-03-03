# NFL Coverage Analysis: Man-to-Man Matchup Prediction System
## Overview

This project implements a machine learning system to analyze
NFL defensive coverage and predict the outcomes of wide receiver
(WR) vs. cornerback (CB) matchups. The system uses player
tracking data from NFL games to classify coverage schemes and
predict play success probabilities.

## Architecture
The project follows a **three-stage pipeline architecture**:
```
Stage 1: Data Cleaning & Feature Engineering
 ↓
Stage 2: Man-to-Man Coverage Classification
 ↓
Stage 3: Matchup Outcome Prediction
 ├── Route Model (detailed route based features)
 ├── Game Model (game context features)
 └── Combined Model (all features)
```
### Pipeline Flow
1. **[Data_Cleaning.ipynb](Data_Cleaning.ipynb)** → Processes
raw tracking data and output a csv with matchup based data `full_cb_wr_data.csv`
2. The files in the feature engineering folder processes `full_cb_wr_data.csv` and creates `final_matchup_data.csv` with rich separation based features
2. **[M2M_model.ipynb](M2M_model.ipynb)** → Classifies man-to-man coverage, adds `pred_man` predictions
3. **Route Model / Combined Model / Game Model** → Predict
offensive win probability for man-to-man plays
---
## File Descriptions
### Core Pipeline Files
#### [Data_Cleaning.ipynb](Data_Cleaning.ipynb)
**Purpose**: Initial data processing and feature engineering
**What it does**:
- Loads weekly tracking data (weeks 1-17) from CSV files
- Merges tracking data with game and play metadata
- Standardizes coordinates and orientations (all plays face
right)
- Standardizes coordinates and orientations (all plays face
right)
- Calculates derived features:
 - Line of scrimmage (LOS) positions
 - Depth from LOS
 - QB positions and orientations
 - Directional vectors (dx, dy)
- Identifies WR-CB matchups by finding the closest CB to each WR
(based on mean distance)
- Computes opponent separation metrics (WR-CB distances)
- Computes teammate separation metrics (CB to nearest defensive
teammate)
- Aggregates frame-by-frame data into per-play summary
statistics:
 - Variance metrics (position, speed)
 - Distance metrics (opponent, sideline, teammate)
 - Orientation metrics (looking at QB, direction differences)
 - Route depth information

#### [M2M_model.ipynb](M2M_model.ipynb)
**Purpose**: Man-to-man coverage classification model
**What it does**:
- Trains an XGBoost binary classifier to predict whether a CB is
in man-to-man coverage
- Uses 35 features including:
 - Positional variance (xVar, yVar)
 - Speed metrics (sVar, sMax)
 - Separation ratios (opponent/teammate distance)
 - Formation features (WR/CB ratio, route type)
- Performs hyperparameter tuning via random search (1000
iterations)
- Achieves ~78% test accuracy


#### [First_Success_Model.ipynb](First_Success_Model.ipynb)
**Purpose**: Initial success prediction model (proof of concept)
**What it does**:
- Predicts whether the offense wins the matchup (`offenseWin`)
- Filters to **targeted routes only** where `pred_man > 0.6`
(high-confidence man coverage)
- Uses 36 base features:
 - All features from data cleaning
 - Play context (down, yards to go, field position)
 - Offensive formation
- Trains XGBoost with random search hyperparameter tuning
- Achieves ~59% test accuracy



#### [Route_model.ipynb](route_model.ipynb)
**Purpose**: Enhanced route-level success prediction with
advanced separation features
**What it does**:
- Improves upon First_Success_Model with additional engineered
features
- Filters to **targeted routes** where `pred_man > 0.3` (lower
threshold = more data)
- Uses 36 features including new separation metrics:
 - `Move Separation`: Separation created by WR's route break
 - `Move Occurred`: Binary flag for whether WR made a move
 - `Time To Move`: Frames elapsed before WR's first move
 - `Release Burst`: WR's acceleration at line of scrimmage
 - `Sep Before Throw (f-2)`: Separation 2 frames before pass
 - `Max/Var/Mean Speed Diff`: Speed differential between WR and
CB
- Achieves ~62% test accuracy
- Identifies route type as most important feature



#### [Game_model.ipynb](game_model.ipynb)
**Purpose**: Simplified game-situation model using only
observable pre-play features
**What it does**:
- Minimalist model using only 16 features:
 - Personnel groupings (WR, TE, RB, CB, FS counts)
 - Formation and down/distance
 - Field position and score
 - Physical matchup data
- Does NOT use tracking-derived features (separation, speed,
etc.)
- Does NOT use tracking-derived features (separation, speed,
etc.)
- Represents what could be predicted before the play runs
- Achieves ~56% test accuracy (baseline performance)
**Output**: Game-context predictions without route/tracking data



#### [Combined_model.ipynb](combined_model.ipynb)
**Purpose**: Comprehensive model combining route features with
game context
**What it does**:
- Combines all features from route model with additional game
situation features:
 - Quarter, down, yards to go, field position
 - Offensive formation, defenders in box, pass rushers
 - Score differential
 - Physical matchup data (height/weight differences)
- Uses 45 total features
- Creates composite feature: `Effective Separation = Sep Before
Throw × WR/CB Ratio`
- Achieves ~60% test accuracy
- Most important features:
 1. Effective Separation
 2. Separation before throw
 3. WR/CB ratio
 4. Height difference
 5. Speed variance



### Feature Engineering Files
#### [feature_engineering/Main_separation_engineering.ipynb]
(feature_engineering/Main_separation_engineering.ipynb)
**Purpose**: Develops the core separation metrics used in models
**What it does**:
- Engineers separation features from frame-by-frame tracking
data
- Calculates 'Move Separation'. A move is defined as a change of direction of 
7.5 degrees from one frame to the next. Move Separation is the separation 3 
frames after the move minus the separation 3 frames before the move.


#### [feature_engineering/Separation_Burst_Training.ipynb]
(feature_engineering/Separation_Burst_Training.ipynb)
**Purpose**: Creates WR release burst and speed differential
features
**What it does**:
- Calculates WR acceleration at snap/release
- These features are integrated into the route and combined
models


#### [feature_engineering/Speed_stats.ipynb]
(feature_engineering/Speed_stats.ipynb)
**Purpose**: Statistical analysis of player speeds
**What it does**:
- Computes relative speed metrics (Max/Mean/Variance of speed
differential)
- These features are integrated into the route and combined
models


### Supplementary Analysis Files
#### [supplementary/Visualizing_moves.ipynb](supplementary/
Visualizing_moves.ipynb)
**Purpose**: Visualization and validation of WR route breaks
**What it does**:
- Creates visualizations of WR movement patterns
- Validates the move detection algorithm
- Shows separation creation on specific plays
- Useful for debugging and understanding feature engineering
---
#### [supplementary/Separation_evaluation.ipynb](supplementary/
Separation_evaluation.ipynb)
**Purpose**: Evaluates separation metrics against outcomes
**What it does**:
- Correlates separation metrics with play success
- Validates that separation features are predictive
- Identifies optimal separation thresholds
---



### Other Files
#### [Animations.ipynb](Animations.ipynb)
**Purpose**: Creates animated visualizations of plays
**What it does**:
- Generates frame-by-frame animations showing player movements
- Visualizes WR-CB matchups over time
- Useful for presentation and validation
---
#### [sample.ipynb](sample.ipynb)
**Purpose**: Exploratory analysis and testing
**What it does**:
- Sandbox for testing the package RouteAnalytics (Main Deliverable)


## Data Files
### Input Data (not in repo)
### Please visit https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data
- `data/week1.csv` through `data/week17.csv`: Weekly tracking
data
- `data/games.csv`: Game metadata
- `data/plays.csv`: Play-level data
### Output Data
- **[final_matchup_data.csv](final_matchup_data.csv)**: Main
dataset with all features and predictions (19,015 WR-CB
matchups)
---
## Model Performance Summary
| Model | Accuracy | Use Case | Key Features |
|-------|----------|----------|--------------|
| **M2M Coverage Model** | 78% | Classify man vs. zone coverage | Separation ratios, movement patterns |
| **Route Model** | 62% | Predict success with route details | Separation metrics, speed differentials |
| **Combined Model** | 60% | Full context prediction | All route + game situation features |
| **Game Model** | 56% | Pre-play prediction | Personnel, formation, down/distance |
---
## Key Findings
### Model Insights
1. **Most Important Features for Success**:
 - Separation before throw (f-2)
 - Route type
 - Variance of the direction of the defender
2. **Coverage Detection**:
 - WR-CB separation ratio to nearest teammate is highly
predictive of man coverage
 - Route variance and looking-at-QB metrics help distinguish
coverage schemes
3. **Matchup Analysis**:
 - Most common matchup: Alshon Jeffery vs Josh Norman (39
plays)
 - Top WRs create more separation and have higher win
probabilities
 - Physical mismatches (height/weight) contribute but are less
important than separation
### Interesting Case Study
- Alshon Jeffery vs Josh Norman:
 - Week 13 average offensive win probability: 41.4%
 - Week 17 average offensive win probability: 46.5%
 - Alshon Jeffery had a much better game in Week 17 
---
## Technologies Used
- **Python 3.x**
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **XGBoost**: Gradient boosting models
- **scikit-learn**: Model evaluation and preprocessing
- **matplotlib/seaborn**: Visualization
---
## Model Training Pipeline
To reproduce the full pipeline:
1. **Data Preparation**:
 ```python
 # Run Data_Cleaning.ipynb
 # Run the feature engineering files to get custom features
 # Creates: final_matchup_data.csv
 ```
2. **Coverage Classification**:
 ```python
 # Run M2M_model.ipynb
 # Adds pred_man column
 ```
3. **Success Prediction** (choose one):
 ```python
 # Option A: Route-focused
 # Run route_model.ipynb
 # Option B: Comprehensive
 # Run combined_model.ipynb
 # Option C: Game-context only
 # Run game_model.ipynb
 ```
---
## Feature Glossary
### Spatial Features
- **xVar, yVar**: Variance in CB's x/y position across frames
- **xLOS**: Line of scrimmage x-coordinate
- **maxDepth**: Maximum depth of WR route from LOS
- **endDepth**: WR depth at final frame
### Speed Features
- **sVar, sMax**: Variance and maximum speed of CB
- **dxVar, dyVar**: Variance in velocity components
- **Max/Mean/Var Speed Diff**: Speed differential metrics
### Separation Features
- **oppDistMean/Var/Max**: Statistics of WR-CB separation
- **mateDistMean/Var/Max**: Statistics of CB to nearest teammate
- **Move Separation**: Change in separation after WR's route
break
- **Sep Before Throw (f-2)**: Separation 2 frames before pass
### Orientation Features
- **dirDiffMean/Var**: Direction difference between WR and CB
- **diffOppMean/Var**: CB orientation vs. direction to WR
- **diffQBMean/Var**: CB orientation vs. direction to QB
- **lookingAtQBMean/Var**: Whether CB is looking at QB
### Context Features
- **WRToCBRatio**: Number of WRs divided by number of CBs
- **defendersInTheBox**: Number of defenders near LOS
- **poss_score_diff**: Score differential (possession team
perspective)
