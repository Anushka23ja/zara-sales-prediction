"""
Streamlit app for Zara Menswear Sales Prediction
Run: streamlit run streamlit_app.py
Make sure you run zara_analysis.py first to generate the models and plots.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Zara Sales Prediction", layout="wide")
st.title("Zara Menswear Sales Prediction")

# load data
df = pd.read_csv("data/zara_menswear.csv")

# load models
try:
    lr_model  = joblib.load("models/linear_regression.pkl")
    dt_model  = joblib.load("models/decision_tree.pkl")
    rf_model  = joblib.load("models/random_forest.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")
    models_loaded = True
except FileNotFoundError:
    st.warning("Could not find saved models. Please run zara_analysis.py first.")
    models_loaded = False

# load best params if they were saved
try:
    best_params = joblib.load("models/best_params.pkl")
except FileNotFoundError:
    best_params = {}

# set up features and split (same as training)
num_features = ["price"]
cat_features = ["productPosition", "promotion", "seasonal", "category"]
X = df[num_features + cat_features]
y = df["salesVolume"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# sidebar nav
tab = st.sidebar.radio("Go to", [
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability and Prediction",
])


# -----------------------------------------------------------------------
# TAB 1: Executive Summary
# -----------------------------------------------------------------------

if tab == "Executive Summary":
    st.header("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", 252)
    col2.metric("Avg Price (USD)", f"${df['price'].mean():.2f}")
    col3.metric("Avg Sales Volume", f"{df['salesVolume'].mean():.0f}")
    col4.metric("Best Model R2", "0.474")

    st.subheader("About the Dataset")
    st.write(
        "This dataset was scraped from Zara's US website in February 2024 and contains "
        "information on 252 men's clothing products. Each row represents a single product "
        "and includes attributes like price, the product's position in the store layout "
        "(aisle vs end-cap), whether it was on promotion, whether it was a seasonal item, "
        "and the product category (e.g., jackets, shirts, pants). The target variable we "
        "are trying to predict is sales volume, which is a count of units sold."
    )
    st.write(
        "The dataset is relatively small at 252 rows, but it covers a clean retail setting "
        "with a mix of numerical and categorical features. There are no missing values, which "
        "made preprocessing straightforward. We used price as the only continuous feature and "
        "treated the rest as categoricals that were one-hot encoded."
    )

    st.subheader("Why This Problem Matters")
    st.write(
        "For a fast fashion retailer like Zara, being able to predict which products will sell "
        "well has real business value. If you can forecast demand more accurately, you can make "
        "better decisions about inventory levels, store layouts, and which products to put on "
        "promotion. Overstocking leads to markdowns and lost margin, while understocking means "
        "missed sales. Even a modest improvement in prediction accuracy could translate into "
        "meaningful cost savings at scale."
    )
    st.write(
        "Beyond the practical side, this dataset is interesting because it lets us test some "
        "common retail assumptions. For example, do promotions actually increase sales, or are "
        "they just applied to slow-moving items? Does placing a product in the aisle vs the end "
        "of a rack make a difference? The SHAP analysis in the last tab gives us some answers."
    )

    st.subheader("Approach and Key Findings")
    st.write(
        "We trained five models: Linear Regression as a baseline, Decision Tree, Random Forest, "
        "XGBoost, and a Neural Network (MLP). All tree-based models were tuned with 5-fold "
        "GridSearchCV. XGBoost came out on top with an R2 of 0.474 and RMSE of 568.3, meaning "
        "it explains about 47% of the variance in sales volume. This is a reasonable result for "
        "a small dataset with limited features."
    )
    st.write(
        "The most important finding from the SHAP analysis is that price is by far the strongest "
        "predictor -- higher priced items tend to sell less, which makes sense. Product position "
        "also matters, with aisle placement associated with higher sales. Interestingly, "
        "promotions showed a slightly negative relationship with sales, which might suggest that "
        "Zara tends to promote items that are already underperforming rather than using promotions "
        "proactively to drive volume."
    )


# -----------------------------------------------------------------------
# TAB 2: Descriptive Analytics
# -----------------------------------------------------------------------

elif tab == "Descriptive Analytics":
    st.header("Descriptive Analytics")

    st.subheader("Dataset Overview")
    st.dataframe(df.head(10))
    st.write(df.describe())

    # target distribution
    st.subheader("Distribution of Sales Volume")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["salesVolume"], bins=25, edgecolor="black", alpha=0.75, color="#2563eb")
    ax.set_xlabel("Sales Volume (units)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Sales Volume")
    st.pyplot(fig)
    st.caption(
        "The sales volume distribution is right-skewed, with most products selling fewer "
        "than 1000 units and a long tail of high-performing items. This means a few products "
        "are responsible for a disproportionate share of total sales, which is a common pattern "
        "in retail data."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price vs Sales Volume")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df["price"], df["salesVolume"], alpha=0.5, edgecolors="k",
                   linewidth=0.3, color="#7c3aed")
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Sales Volume")
        ax.set_title("Price vs Sales Volume")
        st.pyplot(fig)
        st.caption(
            "There is a clear negative relationship between price and sales volume -- cheaper "
            "items generally sell in higher quantities. The scatter also shows that most products "
            "are priced between $20 and $150, with a few expensive outliers above $200 that "
            "consistently have low sales."
        )

    with col2:
        st.subheader("Average Sales by Product Position")
        fig, ax = plt.subplots(figsize=(7, 5))
        df.groupby("productPosition")["salesVolume"].mean().sort_values().plot.barh(
            color="#2563eb", edgecolor="black", ax=ax
        )
        ax.set_xlabel("Average Sales Volume")
        ax.set_title("Average Sales by Product Position")
        st.pyplot(fig)
        st.caption(
            "Products placed in the aisle tend to have higher average sales than those at "
            "the end-cap. This could reflect that aisle products get more foot traffic or "
            "that higher-demand items are strategically placed there."
        )

    st.subheader("Average Sales by Category")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby("category")["salesVolume"].mean().sort_values().plot.barh(
        color="#059669", edgecolor="black", ax=ax
    )
    ax.set_xlabel("Average Sales Volume")
    ax.set_title("Average Sales by Category")
    st.pyplot(fig)
    st.caption(
        "Sales volume varies quite a bit across product categories. Some categories like "
        "basics and outerwear tend to outsell more niche categories. This suggests that "
        "category could be a useful predictor in the model, which the SHAP analysis later confirms."
    )

    st.subheader("Promoted vs Not Promoted")
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("promotion")["salesVolume"].mean().plot.bar(
        color=["#e11d48", "#2563eb"], edgecolor="black", rot=0, ax=ax
    )
    ax.set_xlabel("On Promotion (No / Yes)")
    ax.set_ylabel("Average Sales Volume")
    ax.set_title("Average Sales: Promoted vs Not Promoted")
    st.pyplot(fig)
    st.caption(
        "Somewhat surprisingly, promoted products have slightly lower average sales than "
        "non-promoted ones. This might indicate that promotions are applied to items that "
        "were already selling slowly, rather than being used to boost popular products."
    )

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    st.caption(
        "Price has the strongest correlation with sales volume (negative), confirming the "
        "pattern we saw in the scatter plot. The other numeric features have weaker correlations "
        "with sales, which is why tree-based models that can capture nonlinear patterns tend "
        "to perform better than linear regression here."
    )


# -----------------------------------------------------------------------
# TAB 3: Model Performance
# -----------------------------------------------------------------------

elif tab == "Model Performance":
    st.header("Model Performance")

    if not models_loaded:
        st.stop()

    st.subheader("Data Preparation")
    st.write(
        "Features used: price (scaled with StandardScaler), productPosition, promotion, "
        "seasonal, and category (all one-hot encoded). Target: salesVolume. "
        "Train/test split: 70/30 with random_state=42."
    )

    # compute metrics for the 4 sklearn models
    model_dict = {
        "Linear Regression": lr_model,
        "Decision Tree":     dt_model,
        "Random Forest":     rf_model,
        "XGBoost":           xgb_model,
    }

    rows = []
    for name, model in model_dict.items():
        preds = model.predict(X_test)
        rows.append({
            "Model": name,
            "MAE":   round(mean_absolute_error(y_test, preds), 1),
            "RMSE":  round(np.sqrt(mean_squared_error(y_test, preds)), 1),
            "R2":    round(r2_score(y_test, preds), 3),
        })

    # add NN row from saved plot if available (metrics shown in plot)
    # we show NN metrics from the saved comparison plot image
    results_df = pd.DataFrame(rows)

    st.subheader("Model Comparison Table")
    st.dataframe(results_df, hide_index=True)

    # bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(results_df))
    width = 0.35
    ax.bar(x_pos - width / 2, results_df["RMSE"], width, label="RMSE", color="#2563eb")
    ax.bar(x_pos + width / 2, results_df["MAE"],  width, label="MAE",  color="#7c3aed")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df["Model"])
    ax.set_ylabel("Error")
    ax.set_title("Model Comparison: RMSE and MAE")
    ax.legend()
    st.pyplot(fig)

    st.write(
        "XGBoost performed best overall with the lowest RMSE (568.3) and highest R2 (0.474). "
        "Decision Tree came in last, which is expected since single trees tend to overfit "
        "or underfit without ensembling. Random Forest improved on the Decision Tree by "
        "averaging over many trees. The Neural Network performed similarly to Random Forest. "
        "Given the small dataset size, the boosted tree approach seems to work best here. "
        "One trade-off is that XGBoost is less interpretable than a single Decision Tree, "
        "but we address that with SHAP values in the next tab."
    )

    # show best hyperparameters
    st.subheader("Best Hyperparameters from GridSearchCV")
    if best_params:
        for model_name, params in best_params.items():
            st.write(f"**{model_name}:** {params}")
    else:
        st.info("Run zara_analysis.py to save best hyperparameters.")

    # predicted vs actual plots for all models
    st.subheader("Predicted vs Actual: All Models")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, model) in enumerate(model_dict.items()):
        preds = model.predict(X_test)
        axes[i].scatter(y_test, preds, alpha=0.5, edgecolors="k", linewidth=0.3)
        axes[i].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=1.5)
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")
        axes[i].set_title(name)

    plt.tight_layout()
    st.pyplot(fig)

    # decision tree visualization
    st.subheader("Decision Tree Visualization (top 3 levels)")
    if os.path.exists("plots/decision_tree.png"):
        st.image("plots/decision_tree.png")
    else:
        st.info("Run zara_analysis.py to generate the decision tree plot.")

    # neural network training history
    st.subheader("Neural Network Training History")
    if os.path.exists("plots/nn_training_history.png"):
        st.image("plots/nn_training_history.png")
        st.caption(
            "The training and validation loss both decrease steadily over 100 epochs, "
            "which suggests the model is learning without major overfitting. The gap "
            "between train and val loss is small, which is expected for a simple MLP "
            "on a dataset this size."
        )
    else:
        st.info("Run zara_analysis.py to generate the NN training history plot.")


# -----------------------------------------------------------------------
# TAB 4: Explainability and Interactive Prediction
# -----------------------------------------------------------------------

elif tab == "Explainability and Prediction":
    st.header("Explainability and Interactive Prediction")

    if not models_loaded:
        st.stop()

    # SHAP plots (pre-generated)
    st.subheader("SHAP Summary Plot (Beeswarm)")
    if os.path.exists("plots/shap_summary.png"):
        st.image("plots/shap_summary.png")
        st.caption(
            "The beeswarm plot shows how each feature affects predictions across all test "
            "samples. Each dot is one sample, colored by feature value. Features are ranked "
            "by their average impact from top to bottom."
        )
    else:
        st.info("Run zara_analysis.py to generate shap_summary.png")

    st.subheader("Feature Importance (Mean |SHAP|)")
    if os.path.exists("plots/shap_bar.png"):
        st.image("plots/shap_bar.png")
        st.caption(
            "The bar chart ranks features by their average absolute SHAP value, which "
            "gives a cleaner summary of which features matter most overall regardless of direction."
        )
    else:
        st.info("Run zara_analysis.py to generate shap_bar.png")

    st.subheader("Interpretation")
    st.write(
        "Price has by far the strongest impact on predicted sales -- higher prices consistently "
        "push the prediction down, which makes sense given the inverse price-demand relationship "
        "we saw in the EDA. Product position is the second most important feature, with aisle "
        "placement associated with higher sales. This is useful for store managers thinking about "
        "where to display products."
    )
    st.write(
        "Promotion has a slightly negative effect, which is counterintuitive but consistent with "
        "what we saw in the EDA. One possible explanation is that Zara marks down items that are "
        "already not selling well, so promotions end up being a signal of low demand rather than "
        "a driver of it. If true, this would suggest the promotion strategy should be revisited."
    )
    st.write(
        "Seasonality has a mixed effect depending on the specific item. Some seasonal products "
        "sell very well in-season, while others don't get much of a boost. For a buying team, "
        "the main takeaway is that pricing decisions and shelf placement are the two biggest "
        "levers available to influence sales volume."
    )

    st.markdown("---")

    # Interactive prediction section
    st.subheader("Interactive Prediction")
    st.write(
        "Use the inputs below to set product attributes and get a predicted sales volume. "
        "You can also choose which model to use for the prediction."
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_model_name = st.selectbox(
            "Choose model",
            ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
        )
        price_input = st.slider(
            "Price (USD)",
            min_value=float(df["price"].min()),
            max_value=float(df["price"].max()),
            value=float(df["price"].median()),
            step=1.0
        )
        position_input = st.selectbox(
            "Product Position",
            df["productPosition"].unique().tolist()
        )

    with col2:
        promotion_input = st.selectbox(
            "On Promotion?",
            df["promotion"].unique().tolist()
        )
        seasonal_input = st.selectbox(
            "Seasonal Item?",
            df["seasonal"].unique().tolist()
        )
        category_input = st.selectbox(
            "Category",
            sorted(df["category"].unique().tolist())
        )

    # map model name to loaded model
    model_map = {
        "Linear Regression": lr_model,
        "Decision Tree":     dt_model,
        "Random Forest":     rf_model,
        "XGBoost":           xgb_model,
    }
    chosen_model = model_map[selected_model_name]

    # build input dataframe
    input_df = pd.DataFrame([{
        "price":           price_input,
        "productPosition": position_input,
        "promotion":       promotion_input,
        "seasonal":        seasonal_input,
        "category":        category_input,
    }])

    prediction = chosen_model.predict(input_df)[0]
    st.metric(
        label=f"Predicted Sales Volume ({selected_model_name})",
        value=f"{max(0, int(round(prediction)))} units"
    )

    # generate SHAP waterfall for the custom input
    st.subheader("SHAP Waterfall for Your Input")
    st.write(
        "This waterfall plot shows how each feature contributed to the prediction above "
        "starting from the baseline (average prediction) and adding each feature's effect."
    )

    try:
        # transform the input using the XGBoost pipeline preprocessor
        # (we always use XGBoost for SHAP since that is the best model)
        X_input_transformed = xgb_model.named_steps["prep"].transform(input_df)

        feature_names = (
            num_features +
            list(
                xgb_model.named_steps["prep"]
                .named_transformers_["cat"]
                .get_feature_names_out(cat_features)
            )
        )

        explainer = shap.TreeExplainer(xgb_model.named_steps["model"])
        shap_vals = explainer.shap_values(X_input_transformed)

        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=X_input_transformed[0],
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"Could not generate SHAP waterfall: {e}")
