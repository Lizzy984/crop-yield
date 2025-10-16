# crop_yield_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page config - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌱",
    layout="wide"
)

# Add some CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class CropYieldPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def load_data(self):
        """Load the dataset"""
        try:
            df = pd.read_csv(r"D:\Liz\Documents\crop_yield_dataset.csv")
            return df
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['crop_type', 'region']
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Separate features and target
        X = df_processed.drop(columns=['crop_yield', 'year'])
        y = df_processed['crop_yield']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        numerical_features = ['temperature_avg', 'rainfall_total', 'soil_ph', 
                             'fertilizer_usage', 'sunlight_hours', 'pesticide_usage']
        X_train[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        return X_train, X_test, y_train, y_test, X.columns
    
    def train_model(self, X_train, X_test, y_train, y_test, model_type='Random Forest'):
        """Train the selected model"""
        if model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # Linear Regression
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'model': model,
            'predictions': y_pred,
            'actuals': y_test
        }
        
        # Feature importance for Random Forest
        if model_type == 'Random Forest':
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.model = model
        self.is_trained = True
        
        return metrics

def main():
    # Header
    st.markdown('<div class="main-header">🌱 Crop Yield Predictor - SDG 2: Zero Hunger</div>', unsafe_allow_html=True)
    st.markdown("### Using Machine Learning to Predict Crop Yields for Food Security")
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = CropYieldPredictor()
    
    if 'df' not in st.session_state:
        st.session_state.df = st.session_state.predictor.load_data()
    
    # Check if data loaded successfully
    if st.session_state.df is None:
        st.error("❌ Could not load data file. Please check the file path.")
        return
    
    st.markdown('<div class="success-box">✅ Data loaded successfully! Ready to analyze.</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🌱 Navigation")
    app_mode = st.sidebar.radio(
        "Choose Section:",
        ["📊 Data Overview", "🤖 Train Model", "🔮 Make Predictions", "⚖️ Ethical Analysis"]
    )
    
    # Data Overview Section
    if app_mode == "📊 Data Overview":
        st.header("📊 Dataset Overview")
        
        # Basic info cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(st.session_state.df))
        with col2:
            st.metric("Crop Types", len(st.session_state.df['crop_type'].unique()))
        with col3:
            st.metric("Regions", len(st.session_state.df['region'].unique()))
        with col4:
            st.metric("Avg Yield", f"{st.session_state.df['crop_yield'].mean():.2f} t/ha")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(st.session_state.df, use_container_width=True)
        
        # Statistics
        st.subheader("📈 Basic Statistics")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        # Visualizations
        st.subheader("📊 Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Yield by crop type
            fig, ax = plt.subplots(figsize=(8, 5))
            crop_yield = st.session_state.df.groupby('crop_type')['crop_yield'].mean().sort_values(ascending=False)
            bars = ax.bar(crop_yield.index, crop_yield.values, color=['#2E8B57', '#3CB371', '#90EE90', '#98FB98'])
            ax.set_title('Average Yield by Crop Type', fontweight='bold')
            ax.set_ylabel('Yield (tons/hectare)')
            ax.tick_params(axis='x', rotation=45)
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            st.pyplot(fig)
            
        with col2:
            # Yield by region
            fig, ax = plt.subplots(figsize=(8, 5))
            region_yield = st.session_state.df.groupby('region')['crop_yield'].mean().sort_values(ascending=False)
            bars = ax.bar(region_yield.index, region_yield.values, color=['#4682B4', '#5F9EA0', '#87CEEB', '#B0E0E6'])
            ax.set_title('Average Yield by Region', fontweight='bold')
            ax.set_ylabel('Yield (tons/hectare)')
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            st.pyplot(fig)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Temperature vs Yield
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(st.session_state.df['temperature_avg'], st.session_state.df['crop_yield'], 
                               alpha=0.6, c=st.session_state.df['crop_yield'], cmap='viridis')
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Crop Yield (tons/hectare)')
            ax.set_title('Temperature vs Crop Yield', fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Yield')
            st.pyplot(fig)
            
        with col4:
            # Rainfall vs Yield
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(st.session_state.df['rainfall_total'], st.session_state.df['crop_yield'], 
                               alpha=0.6, c='blue')
            ax.set_xlabel('Rainfall (mm)')
            ax.set_ylabel('Crop Yield (tons/hectare)')
            ax.set_title('Rainfall vs Crop Yield', fontweight='bold')
            st.pyplot(fig)
    
    # Train Model Section
    elif app_mode == "🤖 Train Model":
        st.header("🤖 Train Machine Learning Model")
        
        st.info("""
        **Model Options:**
        - **Random Forest**: Better accuracy, shows feature importance
        - **Linear Regression**: Faster, more interpretable
        """)
        
        model_type = st.selectbox("Select Model Type", ["Random Forest", "Linear Regression"])
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("🔄 Training model... This may take a few seconds."):
                try:
                    X_train, X_test, y_train, y_test, features = st.session_state.predictor.preprocess_data(st.session_state.df)
                    metrics = st.session_state.predictor.train_model(X_train, X_test, y_train, y_test, model_type)
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display metrics
                    st.subheader("📊 Model Performance")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{metrics['r2']:.3f}",
                                 help="How well the model explains the variance in crop yields")
                    with col2:
                        st.metric("MAE", f"{metrics['mae']:.3f}",
                                 help="Mean Absolute Error - Average prediction error")
                    with col3:
                        st.metric("RMSE", f"{metrics['rmse']:.3f}",
                                 help="Root Mean Square Error - Standard deviation of prediction errors")
                    
                    # Plot actual vs predicted
                    st.subheader("📈 Actual vs Predicted Values")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(metrics['actuals'], metrics['predictions'], alpha=0.6, color='#2E8B57')
                    ax.plot([metrics['actuals'].min(), metrics['actuals'].max()], 
                           [metrics['actuals'].min(), metrics['actuals'].max()], 'r--', lw=2, label='Perfect Prediction')
                    ax.set_xlabel('Actual Yield (tons/hectare)')
                    ax.set_ylabel('Predicted Yield (tons/hectare)')
                    ax.set_title('Model Performance: Actual vs Predicted', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Feature importance
                    if st.session_state.predictor.feature_importance is not None:
                        st.subheader("🔍 Feature Importance")
                        st.write("Which factors most influence crop yields?")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        importance_df = st.session_state.predictor.feature_importance.head(10)
                        bars = ax.barh(importance_df['feature'], importance_df['importance'], color='#FF6B6B')
                        ax.set_xlabel('Importance Score')
                        ax.set_title('Top 10 Most Important Features', fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        # Add value labels
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                                   f'{width:.3f}', ha='left', va='center')
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"❌ Error training model: {e}")
    
    # Make Predictions Section
    elif app_mode == "🔮 Make Predictions":
        st.header("🔮 Predict Crop Yield")
        
        if not st.session_state.predictor.is_trained:
            st.warning("⚠️ Please train a model first in the 'Train Model' section.")
            if st.button("Go to Train Model"):
                st.session_state.app_mode = "🤖 Train Model"
                st.rerun()
        else:
            st.success("✅ Model is ready for predictions!")
            
            st.subheader("🌾 Enter Crop Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌡️ Environmental Factors**")
                temperature = st.slider("Average Temperature (°C)", 20.0, 35.0, 25.0, 0.1,
                                       help="Optimal temperature range for crops")
                rainfall = st.slider("Total Rainfall (mm)", 500, 1200, 800, 10,
                                    help="Annual rainfall amount")
                soil_ph = st.slider("Soil pH", 5.5, 8.0, 6.5, 0.1,
                                   help="Soil acidity/alkalinity level")
                
            with col2:
                st.markdown("**🧪 Agricultural Inputs**")
                fertilizer = st.slider("Fertilizer Usage (kg/ha)", 100, 250, 150, 5,
                                      help="Amount of fertilizer applied")
                sunlight = st.slider("Sunlight Hours", 6.0, 8.0, 7.0, 0.1,
                                    help="Daily sunlight exposure")
                pesticide = st.slider("Pesticide Usage", 1.5, 3.0, 2.0, 0.1,
                                     help="Pesticide application rate")
                
            col3, col4 = st.columns(2)
            
            with col3:
                crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Corn", "Soybean"],
                                        help="Select the crop type")
            with col4:
                region = st.selectbox("Region", ["North", "South", "East", "West"],
                                     help="Select the geographical region")
            
            if st.button("🌱 Predict Yield", type="primary", use_container_width=True):
                try:
                    # Prepare input data
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
                    
                    # Create DataFrame
                    input_df = pd.DataFrame([input_data])
                    
                    # Encode categorical variables
                    input_df['crop_type'] = st.session_state.predictor.label_encoders['crop_type'].transform([crop_type])[0]
                    input_df['region'] = st.session_state.predictor.label_encoders['region'].transform([region])[0]
                    
                    # Scale numerical features
                    numerical_features = ['temperature_avg', 'rainfall_total', 'soil_ph', 
                                        'fertilizer_usage', 'sunlight_hours', 'pesticide_usage']
                    input_df[numerical_features] = st.session_state.predictor.scaler.transform(input_df[numerical_features])
                    
                    # Make prediction
                    prediction = st.session_state.predictor.model.predict(input_df)[0]
                    
                    # Display result
                    st.markdown("---")
                    st.markdown(f"<h2 style='text-align: center; color: #2E8B57;'>📊 Prediction Result</h2>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Predicted Yield", 
                            f"{prediction:.2f} tons/hectare",
                            delta=None
                        )
                    
                    # Compare with average
                    avg_yield = st.session_state.df[st.session_state.df['crop_type'] == crop_type]['crop_yield'].mean()
                    diff = prediction - avg_yield
                    diff_percent = (diff / avg_yield) * 100
                    
                    with col2:
                        st.metric(
                            "Compared to Average", 
                            f"{prediction:.2f} tons/hectare",
                            delta=f"{diff:+.2f} ({diff_percent:+.1f}%)",
                            delta_color="normal" if diff >= 0 else "inverse"
                        )
                    
                    # Interpretation
                    st.markdown("---")
                    st.subheader("📝 Interpretation")
                    
                    if diff > 0:
                        st.success(f"**Good news!** This prediction is **{diff_percent:+.1f}% higher** than the average yield for {crop_type}.")
                        st.write("This suggests favorable conditions for crop growth.")
                    else:
                        st.warning(f"**Note:** This prediction is **{diff_percent:+.1f}% lower** than the average yield for {crop_type}.")
                        st.write("Consider optimizing agricultural inputs or environmental conditions.")
                        
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")
    
    # Ethical Considerations Section
    elif app_mode == "⚖️ Ethical Analysis":
        st.header("⚖️ Ethical Considerations & SDG Alignment")
        
        st.markdown("""
        ### 🎯 Alignment with SDG 2: Zero Hunger
        
        This project directly supports **Sustainable Development Goal 2: Zero Hunger** by:
        
        - **🤝 Enhancing Food Security**: Predicting crop yields helps optimize food production
        - **💡 Supporting Farmers**: Data-driven insights assist in resource allocation
        - **🌱 Promoting Sustainability**: Efficient use of water, fertilizers, and pesticides
        - **📊 Enabling Planning**: Better forecasting for food supply chains
        
        ### ⚖️ Key Ethical Considerations
        
        #### 🔍 Data Bias & Representation
        - Ensure diverse representation of farming communities
        - Consider smallholder vs. large-scale farming practices
        - Account for regional variations in climate and soil conditions
        
        #### 🌍 Environmental Impact
        - Balance yield optimization with environmental sustainability
        - Promote responsible use of fertilizers and pesticides
        - Consider long-term soil health and biodiversity
        
        #### 🤝 Fairness & Accessibility
        - Make technology accessible to all farmers, regardless of scale
        - Avoid widening the gap between resource-rich and resource-poor farmers
        - Consider socioeconomic factors in agricultural decision-making
        
        #### 🔒 Transparency & Trust
        - Clearly communicate model limitations and uncertainties
        - Combine ML predictions with local farmer knowledge
        - Ensure interpretability of predictions for end-users
        
        ### 💡 Recommendations for Ethical Implementation
        
        1. **Combine AI with local knowledge** - Blend data-driven insights with farmer expertise
        2. **Regular model updates** - Continuously improve with new data and conditions
        3. **Climate resilience** - Consider climate change impacts on agriculture
        4. **Equitable access** - Ensure technology benefits all stakeholders
        5. **Environmental stewardship** - Promote sustainable farming practices
        
        ### 🌟 Positive Impact Potential
        
        When implemented ethically, this technology can:
        - Increase global food production by 10-20%
        - Reduce resource waste through optimized inputs
        - Support climate-resilient agriculture
        - Empower farmers with data-driven decision making
        """)

if __name__ == "__main__":
    main()