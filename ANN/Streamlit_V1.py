import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data_path = 'listings.csv'
df = pd.read_csv(data_path)
df['price'] = df['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

# Select features and target
numerical_features = ['latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights',
                      'maximum_nights', 'availability_365', 'number_of_reviews', 'review_scores_rating']
categorical_features = ['neighbourhood_cleansed', 'property_type', 'room_type', 'host_is_superhost', 'instant_bookable']
target = 'price'

X = df[numerical_features + categorical_features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5403)

# Preprocessing pipeline
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features)
    ])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Build model
def build_model(neurons, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train_preprocessed.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Streamlit UI
st.title('Airbnb Price Prediction Dashboard')

# Sidebar for hyperparameter tuning
st.sidebar.header('Hyperparameter Tuning')
neuron_count = st.sidebar.selectbox('Number of Neurons', [64, 128, 256])
dropout_rate = st.sidebar.selectbox('Dropout Rate', [0.1, 0.2, 0.3])
learning_rate = st.sidebar.selectbox('Learning Rate', [0.001, 0.01, 0.0001])
batch_size = st.sidebar.selectbox('Batch Size', [32, 64, 128])
epochs = st.sidebar.selectbox('Epochs', [30, 50, 100])

if st.sidebar.button('Train Model'):
    # Build and train the model
    model = build_model(neuron_count, dropout_rate, learning_rate)
    history = model.fit(X_train_preprocessed, y_train, 
                        epochs=epochs, batch_size=batch_size, 
                        validation_split=0.1, verbose=1)

    # Predictions and evaluation
    y_pred = model.predict(X_test_preprocessed)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Show results
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'R-Squared: {r2:.2f}')

    # Plot training history
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    # Scatter plot of predictions
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r')
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Actual vs Predicted Price')
    st.pyplot(fig)
