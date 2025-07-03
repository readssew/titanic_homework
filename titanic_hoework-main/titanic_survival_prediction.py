"""
Titanic Survival Prediction
This script predicts passenger survival on the Titanic using machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the Titanic dataset"""
    print("Loading Titanic dataset...")
    df = pd.read_csv('train.csv')
    
    print("\n=== Dataset Information ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nSurvival rate: {df['Survived'].mean():.2%}")
    
    return df

def preprocess_data(df):
    """Clean and preprocess the data"""
    print("\n=== Data Preprocessing ===")
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Create new features
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Simplify titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'].fillna('Rare', inplace=True)
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    le_title = LabelEncoder()
    
    data['Sex_encoded'] = le_sex.fit_transform(data['Sex'])
    data['Embarked_encoded'] = le_embarked.fit_transform(data['Embarked'])
    data['Title_encoded'] = le_title.fit_transform(data['Title'])
    
    # Select features for modeling
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 
                'Embarked_encoded', 'FamilySize', 'IsAlone', 'Title_encoded']
    
    X = data[features]
    y = data['Survived']
    
    print(f"Selected features: {features}")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y, data

def create_visualizations(data):
    """Create comprehensive visualizations"""
    print("\n=== Creating Visualizations ===")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Titanic Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Survival rate by gender
    survival_by_sex = data.groupby('Sex')['Survived'].mean()
    axes[0, 0].bar(survival_by_sex.index, survival_by_sex.values, color=['lightcoral', 'lightblue'])
    axes[0, 0].set_title('Survival Rate by Gender')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(survival_by_sex.values):
        axes[0, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # 2. Survival rate by class
    survival_by_class = data.groupby('Pclass')['Survived'].mean()
    axes[0, 1].bar(survival_by_class.index, survival_by_class.values, color=['gold', 'orange', 'red'])
    axes[0, 1].set_title('Survival Rate by Passenger Class')
    axes[0, 1].set_xlabel('Passenger Class')
    axes[0, 1].set_ylabel('Survival Rate')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(survival_by_class.values):
        axes[0, 1].text(i+1, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # 3. Age distribution by survival
    data[data['Survived'] == 0]['Age'].hist(bins=30, alpha=0.7, label='Not Survived', ax=axes[0, 2], color='red')
    data[data['Survived'] == 1]['Age'].hist(bins=30, alpha=0.7, label='Survived', ax=axes[0, 2], color='green')
    axes[0, 2].set_title('Age Distribution by Survival')
    axes[0, 2].set_xlabel('Age')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    
    # 4. Fare distribution by survival
    data[data['Survived'] == 0]['Fare'].hist(bins=30, alpha=0.7, label='Not Survived', ax=axes[1, 0], color='red')
    data[data['Survived'] == 1]['Fare'].hist(bins=30, alpha=0.7, label='Survived', ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Fare Distribution by Survival')
    axes[1, 0].set_xlabel('Fare')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 200)
    
    # 5. Family size vs survival
    family_survival = data.groupby('FamilySize')['Survived'].mean()
    axes[1, 1].plot(family_survival.index, family_survival.values, marker='o', linewidth=2, markersize=8)
    axes[1, 1].set_title('Survival Rate by Family Size')
    axes[1, 1].set_xlabel('Family Size')
    axes[1, 1].set_ylabel('Survival Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Correlation heatmap
    numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    correlation_matrix = data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
    axes[1, 2].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('titanic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'titanic_analysis.png'")

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    print("\n=== Model Training and Evaluation ===")
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machine': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    print("Training models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'Support Vector Machine':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred))
    
    return results, scaler

def create_results_visualization(results, y_test):
    """Create visualization of model results"""
    print("\n=== Creating Results Visualization ===")
    
    # Create results comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    bars = axes[0, 0].bar(model_names, accuracies, color=['lightgreen', 'lightblue', 'lightcoral'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}', ha='center', fontweight='bold')
    
    # 2. Confusion matrix for best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_predictions = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. ROC-like plot showing prediction probabilities
    best_probabilities = results[best_model_name]['probabilities']
    survived_probs = best_probabilities[y_test == 1]
    not_survived_probs = best_probabilities[y_test == 0]
    
    axes[1, 0].hist(not_survived_probs, bins=20, alpha=0.7, label='Not Survived', color='red', density=True)
    axes[1, 0].hist(survived_probs, bins=20, alpha=0.7, label='Survived', color='green', density=True)
    axes[1, 0].set_title(f'Prediction Probability Distribution - {best_model_name}')
    axes[1, 0].set_xlabel('Survival Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    
    # 4. Feature importance (for Random Forest)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                        'Embarked', 'FamilySize', 'IsAlone', 'Title']
        importances = rf_model.feature_importances_
        
        # Sort features by importance
        feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        features, importance_values = zip(*feature_importance)
        
        axes[1, 1].barh(range(len(features)), importance_values)
        axes[1, 1].set_yticks(range(len(features)))
        axes[1, 1].set_yticklabels(features)
        axes[1, 1].set_title('Feature Importance (Random Forest)')
        axes[1, 1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results visualization saved as 'model_results.png'")
    print(f"\nBest performing model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")

def main():
    """Main function to run the entire pipeline"""
    print("=" * 50)
    print("TITANIC SURVIVAL PREDICTION")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Preprocess data
    X, y, processed_data = preprocess_data(df)
    
    # Create visualizations
    create_visualizations(processed_data)
    
    # Split data into train and test sets
    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Training survival rate: {y_train.mean():.2%}")
    print(f"Test survival rate: {y_test.mean():.2%}")
    
    # Train models
    results, scaler = train_models(X_train, X_test, y_train, y_test)
    
    # Create results visualization
    create_results_visualization(results, y_test)
    
    print("\n=== Analysis Complete ===")
    print("Files created:")
    print("- titanic_analysis.png: Dataset exploration visualizations")
    print("- model_results.png: Model performance comparison")

if __name__ == "__main__":
    main()
