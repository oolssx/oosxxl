import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import shap
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CreditScorePredictor:
    def __init__(self, model_path=None):
        self.gbdt_model = None
        self.lstm_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.feature_names = self._initialize_feature_names()
        self.model_weights = {'gbdt': 0.45, 'lstm': 0.35, 'rf': 0.20}
        
        if model_path:
            self.load_models(model_path)
        else:
            self._build_models()
    
    def _initialize_feature_names(self):
        return [
            'credit_history_months', 'total_accounts', 'active_accounts', 'closed_accounts',
            'total_credit_limit', 'total_credit_used', 'credit_utilization_ratio',
            'num_credit_cards', 'num_loans', 'num_mortgages',
            'total_overdue_count', 'max_overdue_days', 'recent_overdue_count',
            'hard_inquiries_6m', 'hard_inquiries_12m', 'soft_inquiries_6m',
            'total_debt', 'monthly_payment', 'debt_to_income_ratio',
            'avg_account_age_months', 'oldest_account_age_months', 'newest_account_age_months',
            'num_delinquent_accounts', 'public_records_count', 'collections_count',
            'total_balance', 'revolving_balance', 'installment_balance',
            'mortgage_balance', 'auto_loan_balance', 'student_loan_balance',
            'payment_history_score', 'account_mix_score', 'new_credit_score',
            'available_credit', 'total_limits', 'bankcard_utilization',
            'revolving_utilization', 'num_satisfactory_accounts', 'num_accounts_30dpd',
            'num_accounts_60dpd', 'num_accounts_90dpd', 'inquiries_last_month',
            'inquiries_last_3months', 'inquiries_last_6months', 'accounts_opened_24m',
            'months_since_oldest_trade', 'months_since_recent_trade', 'months_since_recent_inquiry',
            'num_credit_mix_types', 'average_months_in_file'
        ]
    
    def _build_models(self):
        self.gbdt_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            max_features='sqrt',
            random_state=42,
            loss='huber',
            alpha=0.9
        )
        
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.lstm_model = self._build_lstm_architecture()
    
    def _build_lstm_architecture(self):
        sequence_input = Input(shape=(24, 30), name='sequence_input')
        
        lstm_out = LSTM(128, return_sequences=True)(sequence_input)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        
        lstm_out = LSTM(64, return_sequences=True)(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        
        lstm_out = LSTM(32)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
        
        static_input = Input(shape=(len(self.feature_names),), name='static_input')
        static_dense = Dense(64, activation='relu')(static_input)
        static_dense = Dropout(0.2)(static_dense)
        static_dense = Dense(32, activation='relu')(static_dense)
        
        merged = Concatenate()([lstm_out, static_dense])
        merged = Dense(64, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(32, activation='relu')(merged)
        merged = Dense(16, activation='relu')(merged)
        
        output = Dense(1, name='score_output')(merged)
        
        model = Model(inputs=[sequence_input, static_input], outputs=output)
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_features(self, user_data):
        features = []
        
        for feature_name in self.feature_names:
            if feature_name in user_data:
                features.append(user_data[feature_name])
            else:
                features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def prepare_sequence_data(self, user_history):
        if len(user_history) < 24:
            padding_length = 24 - len(user_history)
            padding = np.zeros((padding_length, 30))
            sequence = np.vstack([padding, user_history])
        else:
            sequence = user_history[-24:]
        
        return sequence.reshape(1, 24, 30)
    
    def train(self, X_train, y_train, X_sequence_train, validation_split=0.2):
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        X_seq_train_split, X_seq_val_split = train_test_split(
            X_sequence_train, test_size=validation_split, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train_split)
        X_val_scaled = self.scaler.transform(X_val_split)
        
        print("Training GBDT model...")
        self.gbdt_model.fit(X_train_scaled, y_train_split)
        gbdt_val_score = self.gbdt_model.score(X_val_scaled, y_val_split)
        print(f"GBDT R² Score: {gbdt_val_score:.4f}")
        
        print("Training Random Forest model...")
        self.rf_model.fit(X_train_scaled, y_train_split)
        rf_val_score = self.rf_model.score(X_val_scaled, y_val_split)
        print(f"Random Forest R² Score: {rf_val_score:.4f}")
        
        print("Training LSTM model...")
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001
        )
        
        history = self.lstm_model.fit(
            [X_seq_train_split, X_train_scaled],
            y_train_split,
            validation_data=([X_seq_val_split, X_val_scaled], y_val_split),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, user_features, user_history):
        features_array = self.prepare_features(user_features)
        features_scaled = self.scaler.transform(features_array)
        
        gbdt_prediction = self.gbdt_model.predict(features_scaled)[0]
        
        rf_prediction = self.rf_model.predict(features_scaled)[0]
        
        sequence_data = self.prepare_sequence_data(user_history)
        lstm_prediction = self.lstm_model.predict(
            [sequence_data, features_scaled],
            verbose=0
        )[0][0]
        
        ensemble_prediction = (
            self.model_weights['gbdt'] * gbdt_prediction +
            self.model_weights['lstm'] * lstm_prediction +
            self.model_weights['rf'] * rf_prediction
        )
        
        predictions_array = np.array([gbdt_prediction, lstm_prediction, rf_prediction])
        std_dev = np.std(predictions_array)
        confidence = max(0, 1 - (std_dev / 100))
        
        confidence_interval = (
            ensemble_prediction - 1.96 * std_dev,
            ensemble_prediction + 1.96 * std_dev
        )
        
        return {
            'score': float(ensemble_prediction),
            'confidence': float(confidence),
            'confidence_interval': confidence_interval,
            'component_scores': {
                'gbdt': float(gbdt_prediction),
                'lstm': float(lstm_prediction),
                'rf': float(rf_prediction)
            }
        }
    
    def explain_prediction(self, user_features):
        features_array = self.prepare_features(user_features)
        features_scaled = self.scaler.transform(features_array)
        
        explainer = shap.TreeExplainer(self.gbdt_model)
        shap_values = explainer.shap_values(features_scaled)
        
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_importance[feature_name] = float(shap_values[0][i])
        
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            'feature_importance': dict(sorted_importance[:15]),
            'base_value': float(explainer.expected_value),
            'prediction_value': float(self.gbdt_model.predict(features_scaled)[0])
        }
    
    def simulate_scenario(self, user_features, user_history, changes):
        original_prediction = self.predict(user_features, user_history)
        
        modified_features = user_features.copy()
        for feature, change_value in changes.items():
            if feature in modified_features:
                if isinstance(change_value, dict):
                    if change_value['type'] == 'absolute':
                        modified_features[feature] = change_value['value']
                    elif change_value['type'] == 'relative':
                        modified_features[feature] *= (1 + change_value['value'])
                else:
                    modified_features[feature] = change_value
        
        new_prediction = self.predict(modified_features, user_history)
        
        return {
            'original_score': original_prediction['score'],
            'new_score': new_prediction['score'],
            'score_change': new_prediction['score'] - original_prediction['score'],
            'original_confidence': original_prediction['confidence'],
            'new_confidence': new_prediction['confidence'],
            'modified_features': changes
        }
    
    def batch_predict(self, users_data):
        results = []
        
        for user_data in users_data:
            try:
                prediction = self.predict(
                    user_data['features'],
                    user_data['history']
                )
                results.append({
                    'user_id': user_data.get('user_id', 'unknown'),
                    'prediction': prediction,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'user_id': user_data.get('user_id', 'unknown'),
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
    
    def calculate_feature_trends(self, user_history_sequence):
        trends = {}
        
        if len(user_history_sequence) < 2:
            return trends
        
        recent_data = user_history_sequence[-6:]
        
        for i in range(recent_data.shape[1]):
            values = recent_data[:, i]
            if len(values) >= 2:
                trend = np.polyfit(range(len(values)), values, 1)[0]
                trends[f'feature_{i}_trend'] = float(trend)
        
        return trends
    
    def evaluate_credit_risk_level(self, credit_score):
        if credit_score >= 750:
            return 'excellent', 'Very Low Risk'
        elif credit_score >= 700:
            return 'good', 'Low Risk'
        elif credit_score >= 650:
            return 'fair', 'Medium Risk'
        elif credit_score >= 600:
            return 'poor', 'High Risk'
        else:
            return 'very_poor', 'Very High Risk'
    
    def save_models(self, path_prefix):
        joblib.dump(self.gbdt_model, f'{path_prefix}_gbdt.pkl')
        joblib.dump(self.rf_model, f'{path_prefix}_rf.pkl')
        joblib.dump(self.scaler, f'{path_prefix}_scaler.pkl')
        self.lstm_model.save(f'{path_prefix}_lstm.h5')
        
        print(f"Models saved to {path_prefix}")
    
    def load_models(self, path_prefix):
        self.gbdt_model = joblib.load(f'{path_prefix}_gbdt.pkl')
        self.rf_model = joblib.load(f'{path_prefix}_rf.pkl')
        self.scaler = joblib.load(f'{path_prefix}_scaler.pkl')
        self.lstm_model = tf.keras.models.load_model(f'{path_prefix}_lstm.h5')
        
        print(f"Models loaded from {path_prefix}")


def generate_synthetic_data(n_samples=10000):
    np.random.seed(42)
    
    credit_history = np.random.randint(6, 300, n_samples)
    total_accounts = np.random.randint(1, 30, n_samples)
    credit_limit = np.random.uniform(5000, 100000, n_samples)
    credit_used = credit_limit * np.random.uniform(0.1, 0.9, n_samples)
    
    data = {
        'credit_history_months': credit_history,
        'total_accounts': total_accounts,
        'total_credit_limit': credit_limit,
        'total_credit_used': credit_used,
        'credit_utilization_ratio': credit_used / credit_limit,
        'num_credit_cards': np.random.randint(1, 10, n_samples),
        'total_overdue_count': np.random.randint(0, 5, n_samples),
        'hard_inquiries_6m': np.random.randint(0, 8, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    base_score = 500
    score = base_score + (credit_history / 3) - (data['total_overdue_count'] * 30)
    score -= (data['credit_utilization_ratio'] * 100)
    score -= (data['hard_inquiries_6m'] * 10)
    score += (total_accounts * 2)
    
    score = np.clip(score + np.random.normal(0, 20, n_samples), 300, 850)
    
    return df, score