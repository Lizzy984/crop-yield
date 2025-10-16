# Main app for predicting crop yields
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set up the page
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌱",
    layout="wide"
)

# Make it look nice
st.markdown("""
<style>
    .big-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .green-box {
        background-color: #f0fff0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class CropPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.trained = False
        
    def load_data(self):
        """Get our crop data from the file"""
        try:
            # This looks for the data file in the same folder
            df = pd.read_csv("crop_yield_dataset.csv")
            return df
        except Exception as e:
            st.error(f"Oops! Couldn't load the data: {e}")
            return None
    
    def prepare_data(self, df):
        """Get the data ready for the computer to learn from"""
        df_clean = df.copy()
        
        # Convert crop names and regions to numbers
        categorical_cols = ['crop_type', 'region']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df_clean[col] = self.label_encoders[col].fit_transform(df_clean[col])
        
        # Separate what we're trying to predict from what we're using to predict
        X = df_clean.drop(columns=['crop_yield', 'year'])
        y = df_clean['crop_yield']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the numbers so they're easier for the computer to work with
        number_cols = ['temperature_avg', 'rainfall_total', 'soil_ph', 
                      'fertilizer_usage', 'sunlight_hours', 'pesticide_usage']
        X_train[number_cols] = self.scaler.fit_transform(X_train[number_cols])
        X_test[number_cols] = self.scaler.transform(X_test[number_cols])
        
        return X_train, X_test, y_train, y_test, X.columns
    
    def train_model(self, X_train, X_test, y_train, y_test, model_type='Random Forest'):
        """Teach the computer to predict crop yields"""
        if model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        
        # See how well it learned
        y_pred = model.predict(X_test)
        
        results = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'model': model,
            'predictions': y_pred,
            'actuals': y_test
        }
        
        # For Random Forest, see which factors matter most
        if model_type == 'Random Forest':
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.model = model
        self.trained = True
        
        return results

def main():
    # Show the main title
    st.markdown('<div class="big-header">🌱 Crop Yield Predictor</div>', unsafe_allow_html=True)
    st.markdown("### Helping farmers grow more food using data")
    
    # Set up our predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = CropPredictor()
    
    if 'df' not in st.session_state:
        st.session_state.df = st.session_state.predictor.load_data()
    
    # Check if we got the data
    if st.session_state.df is None:
        st.error("Couldn't find the data file. Make sure it's in the same folder.")
        return
    
    st.markdown('<div class="green-box">✅ Great! We found your crop data.</div>', unsafe_allow_html=True)
    
    # Let people choose what to do
    st.sidebar.title("What do you want to do?")
    page = st.sidebar.radio(
        "Choose:",
        ["Look at the Data", "Train the Model", "Make Predictions", "Why This Matters"]
    )
    
    # Show the data
    if page == "Look at the Data":
        st.header("What's in our data?")
        
        # Quick facts
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(st.session_state.df))
        with col2:
            st.metric("Crop Types", len(st.session_state.df['crop_type'].unique()))
        with col3:
            st.metric("Regions", len(st.session_state.df['region'].unique()))
        with col4:
            st.metric("Avg Yield", f"{st.session_state.df['crop_yield'].mean():.2f} tons/hectare")
        
        # Show the actual data
        st.subheader("The raw data")
        st.dataframe(st.session_state.df, use_container_width=True)
        
        # Show some statistics
        st.subheader("Number summary")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        # Show some charts
        st.subheader("What the data tells us")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Which crops give the best yields?
            fig, ax = plt.subplots(figsize=(8, 5))
            crop_yield = st.session_state.df.groupby('crop_type')['crop_yield'].mean().sort_values(ascending=False)
            bars = ax.bar(crop_yield.index, crop_yield.values, color=['#2E8B57', '#3CB371', '#90EE90', '#98FB98'])
            ax.set_title('Which crops give the best yields?')
            ax.set_ylabel('Yield (tons/hectare)')
            ax.tick_params(axis='x', rotation=45)
            # Put numbers on the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            st.pyplot(fig)
            
        with col2:
            # Which regions are best for farming?
            fig, ax = plt.subplots(figsize=(8, 5))
            region_yield = st.session_state.df.groupby('region')['crop_yield'].mean().sort_values(ascending=False)
            bars = ax.bar(region_yield.index, region_yield.values, color=['#4682B4', '#5F9EA0', '#87CEEB', '#B0E0E6'])
            ax.set_title('Which regions have the best yields?')
            ax.set_ylabel('Yield (tons/hectare)')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            st.pyplot(fig)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # How does temperature affect yield?
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(st.session_state.df['temperature_avg'], st.session_state.df['crop_yield'], alpha=0.6)
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Crop Yield')
            ax.set_title('Does temperature affect yield?')
            st.pyplot(fig)
            
        with col4:
            # How does rainfall affect yield?
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(st.session_state.df['rainfall_total'], st.session_state.df['crop_yield'], alpha=0.6, color='blue')
            ax.set_xlabel('Rainfall (mm)')
            ax.set_ylabel('Crop Yield')
            ax.set_title('Does rainfall affect yield?')
            st.pyplot(fig)
    
    # Train the model
    elif page == "Train the Model":
        st.header("Teach the computer to predict yields")
        
        st.info("""
        **How this works:**
        - **Random Forest**: More accurate, shows what factors matter most
        - **Linear Regression**: Simpler, easier to understand
        """)
        
        model_choice = st.selectbox("Pick a method:", ["Random Forest", "Linear Regression"])
        
        if st.button("Train the Model", type="primary", use_container_width=True):
            with st.spinner("Teaching the computer... this might take a moment"):
                try:
                    X_train, X_test, y_train, y_test, features = st.session_state.predictor.prepare_data(st.session_state.df)
                    results = st.session_state.predictor.train_model(X_train, X_test, y_train, y_test, model_choice)
                    
                    st.success("Nice! The computer learned how to predict yields!")
                    
                    # Show how well it did
                    st.subheader("How well did it learn?")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy Score", f"{results['r2']:.3f}",
                                 help="How close the predictions are to reality (0-1 scale)")
                    with col2:
                        st.metric("Average Error", f"{results['mae']:.3f}",
                                 help="How far off the predictions are on average")
                    with col3:
                        st.metric("Error Range", f"{results['rmse']:.3f}",
                                 help="How spread out the errors are")
                    
                    # Show predictions vs reality
                    st.subheader("Predictions vs Actual Yields")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(results['actuals'], results['predictions'], alpha=0.6, color='#2E8B57')
                    ax.plot([results['actuals'].min(), results['actuals'].max()], 
                           [results['actuals'].min(), results['actuals'].max()], 'r--', lw=2, label='Perfect Prediction')
                    ax.set_xlabel('Actual Yield (tons/hectare)')
                    ax.set_ylabel('Predicted Yield (tons/hectare)')
                    ax.set_title('How close are our predictions?')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Show what matters most
                    if st.session_state.predictor.feature_importance is not None:
                        st.subheader("What affects yields the most?")
                        st.write("The computer figured out which factors are most important:")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        importance_df = st.session_state.predictor.feature_importance.head(10)
                        bars = ax.barh(importance_df['feature'], importance_df['importance'], color='#FF6B6B')
                        ax.set_xlabel('Importance')
                        ax.set_title('Top factors that influence crop yields')
                        ax.grid(True, alpha=0.3)
                        # Add numbers
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                                   f'{width:.3f}', ha='left', va='center')
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
    
    # Make predictions
    elif page == "Make Predictions":
        st.header("Predict crop yields")
        
        if not st.session_state.predictor.trained:
            st.warning("You need to train the model first! Go to the 'Train the Model' section.")
            if st.button("Go Train the Model"):
                st.session_state.page = "Train the Model"
                st.rerun()
        else:
            st.success("Ready to make predictions!")
            
            st.subheader("Enter your farm conditions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Weather & Soil**")
                temperature = st.slider("Average Temperature (°C)", 20.0, 35.0, 25.0, 0.1)
                rainfall = st.slider("Total Rainfall (mm)", 500, 1200, 800, 10)
                soil_ph = st.slider("Soil pH", 5.5, 8.0, 6.5, 0.1)
                
            with col2:
                st.write("**Farm Inputs**")
                fertilizer = st.slider("Fertilizer (kg/ha)", 100, 250, 150, 5)
                sunlight = st.slider("Sunlight Hours", 6.0, 8.0, 7.0, 0.1)
                pesticide = st.slider("Pesticide", 1.5, 3.0, 2.0, 0.1)
                
            col3, col4 = st.columns(2)
            
            with col3:
                crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Corn", "Soybean"])
            with col4:
                region = st.selectbox("Region", ["North", "South", "East", "West"])
            
            if st.button("Predict My Yield", type="primary", use_container_width=True):
                try:
                    # Prepare the input
                    input_data = {
                        'temperature_avg': temperature,
                        'rainfall_total': rainfall,
                        'soil_ph': soil_ph,
                        'fertilizer_usage': fertilizer,
                        'sunlight_hours': sunlight,
                        'pesticide_usage': pesticide,
                        'crop_type': crop_type,
                        'region': region
                    }
                    
                    # Convert to the format the computer expects
                    input_df = pd.DataFrame([input_data])
                    input_df['crop_type'] = st.session_state.predictor.label_encoders['crop_type'].transform([crop_type])[0]
                    input_df['region'] = st.session_state.predictor.label_encoders['region'].transform([region])[0]
                    
                    # Scale the numbers
                    number_cols = ['temperature_avg', 'rainfall_total', 'soil_ph', 
                                  'fertilizer_usage', 'sunlight_hours', 'pesticide_usage']
                    input_df[number_cols] = st.session_state.predictor.scaler.transform(input_df[number_cols])
                    
                    # Make the prediction
                    prediction = st.session_state.predictor.model.predict(input_df)[0]
                    
                    # Show the result
                    st.markdown("---")
                    st.markdown(f"<h2 style='text-align: center; color: #2E8B57;'>📊 Your Prediction</h2>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Expected Yield", 
                            f"{prediction:.2f} tons/hectare"
                        )
                    
                    # Compare with average
                    avg_yield = st.session_state.df[st.session_state.df['crop_type'] == crop_type]['crop_yield'].mean()
                    diff = prediction - avg_yield
                    diff_percent = (diff / avg_yield) * 100
                    
                    with col2:
                        st.metric(
                            "Compared to Average", 
                            f"{prediction:.2f} tons/hectare",
                            delta=f"{diff:+.2f} ({diff_percent:+.1f}%)"
                        )
                    
                    # Give some advice
                    st.markdown("---")
                    st.subheader("What this means for you:")
                    
                    if diff > 0:
                        st.success(f"**Good news!** Your predicted yield is **{diff_percent:+.1f}% higher** than average for {crop_type}.")
                        st.write("Your farm conditions look really favorable for this crop!")
                    else:
                        st.warning(f"**Heads up:** Your predicted yield is **{diff_percent:+.1f}% lower** than average for {crop_type}.")
                        st.write("You might want to adjust your farming practices or try a different crop.")
                        
                except Exception as e:
                    st.error(f"Something went wrong with the prediction: {e}")
    
    # Why this matters
    elif page == "Why This Matters":
        st.header("Why we built this")
        
        st.markdown("""
        ### Fighting Hunger with Data
        
        This project is part of a bigger mission: **ending world hunger**. 
        
        **How predicting crop yields helps:**
        
        - **Farmers grow more food** by knowing the best conditions
        - **Less waste** of water, fertilizer, and pesticides
        - **Better planning** for food supplies in communities
        - **Helping small farmers** compete with big farms
        
        ### Things we thought about:
        
        **Making it fair for everyone:**
        - Works for different types of farms (big and small)
        - Considers different regions and soil types
        - Doesn't favor any particular group
        
        **Keeping it real:**
        - We show how the predictions work
        - We're honest about what the computer can and can't do
        - We combine computer smarts with farmer knowledge
        
        **Good for the planet:**
        - Helps use water and fertilizer wisely
        - Reduces chemical runoff
        - Supports sustainable farming
        
        ### The bottom line:
        
        When farmers can grow more food efficiently, everyone wins. 
        This tool is one small piece of solving the big hunger problem.
        """)

if __name__ == "__main__":
    main()
