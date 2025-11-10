import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os
import ast
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
class StateOfArtModel:
    def __init__(self, train_dir, val_dir, test_dir, output_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sequence_length = 120
        self.overlap = 0.75
        self.batch_size = 16
        self.ear_features = 3
        self.coordinate_features = 468 * 2
        self.temporal_dim = self.sequence_length
        self.spatial_height = 25
        self.spatial_width = 40
        self.channels = 1
        self.tlx_targets = ['NASA_mental', 'NASA_temporal', 'NASA_effort', 
                           'NASA_performance', 'NASA_frustration']
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        self.model = None
        self.history = None
        self.tlx_data = None
    
    def load_tlx_data(self):
        print("📊 Loading and preparing TLX data...")
        possible_tlx_paths = [
            os.path.join(self.train_dir, 'tlx.csv'),
            os.path.join(os.path.dirname(self.train_dir), 'tlx.csv'),
            os.path.join(self.train_dir, '..', 'tlx.csv'),
            '/workspace/Jaynab/split/tlx.csv',
            '/workspace/Jaynab/tlx.csv'
        ]
        tlx_df = None
        for tlx_path in possible_tlx_paths:
            try:
                tlx_df = pd.read_csv(tlx_path)
                print(f"✅ Found TLX data at: {tlx_path}")
                break
            except FileNotFoundError:
                continue
        if tlx_df is None:
            raise FileNotFoundError("Could not find TLX CSV file in any expected location")
        print(f"TLX data shape: {tlx_df.shape}")
        print(f"TLX columns: {list(tlx_df.columns)}")
        if 'condition' not in tlx_df.columns:
            print("🔧 Creating condition mapping from TLX structure...")
            condition_cols = []
            for col in tlx_df.columns:
                if any(x in col.lower() for x in ['participant', 'camera', 'speed', 'p', 'cam']):
                    condition_cols.append(col)
            print(f"Found condition-related columns: {condition_cols}")
            if len(condition_cols) >= 3:
                conditions = []
                for idx, row in tlx_df.iterrows():
                    participant = None
                    camera = None
                    speed = None
                    for col in tlx_df.columns:
                        if 'participant' in col.lower() or col.lower().startswith('p'):
                            participant = str(row[col]).replace('P', '').replace('p', '')
                        elif 'camera' in col.lower() or 'cam' in col.lower():
                            camera = str(row[col])
                        elif 'speed' in col.lower():
                            speed = str(row[col])
                    if participant and camera and speed:
                        condition = f"P{participant}_cam_{camera}_speed_{speed}"
                        conditions.append(condition)
                    else:
                        conditions.append(f"condition_{idx}")
                tlx_df['condition'] = conditions
            else:
                tlx_df['condition'] = [f"condition_{i}" for i in range(len(tlx_df))]
        self.tlx_data = tlx_df
        print(f"✅ TLX data prepared with {len(tlx_df)} conditions")
        return tlx_df
    
    def parse_coordinate_string(self, coord_string):
        try:
            if isinstance(coord_string, str):
                coord_list = ast.literal_eval(coord_string)
                return np.array(coord_list, dtype=np.float32)
            else:
                return np.array([coord_string], dtype=np.float32)
        except:
            return np.zeros(468, dtype=np.float32)
    
    def parse_ear_string(self, ear_string):
        try:
            if isinstance(ear_string, str):
                ear_list = ast.literal_eval(ear_string)
                return float(ear_list[0]) if isinstance(ear_list, list) else float(ear_list)
            else:
                return float(ear_string)
        except:
            return 0.0
    
    def extract_cognitive_features(self, coords_x, coords_y):
        try:
            if len(coords_x) < 468 or len(coords_y) < 468:
                return np.zeros(15, dtype=np.float32)
            coords = np.column_stack([coords_x[:468], coords_y[:468]])
            face_center_indices = [1, 2, 5, 6, 9, 10]
            if all(i < len(coords) for i in face_center_indices):
                face_center = np.mean(coords[face_center_indices], axis=0)
            else:
                face_center = np.mean(coords[:10], axis=0)
            left_eye_indices = list(range(33, min(42, len(coords))))
            right_eye_indices = list(range(263, min(272, len(coords))))
            left_eye_coords = coords[left_eye_indices] if left_eye_indices else coords[:5]
            right_eye_coords = coords[right_eye_indices] if right_eye_indices else coords[5:10]
            left_eye_center = np.mean(left_eye_coords, axis=0)
            right_eye_center = np.mean(right_eye_coords, axis=0)
            eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
            features = [
                eye_distance,
                np.std(coords[:, 0]),
                np.std(coords[:, 1]),
                np.mean(coords[:, 0]),
                np.mean(coords[:, 1]),
                np.linalg.norm(left_eye_center - face_center),
                np.linalg.norm(right_eye_center - face_center),
                np.max(coords[:, 0]) - np.min(coords[:, 0]),
                np.max(coords[:, 1]) - np.min(coords[:, 1]),
                np.std(left_eye_coords.flatten()),
                np.std(right_eye_coords.flatten()),
                np.mean(coords[:20, 0]),
                np.mean(coords[:20, 1]),
                np.mean(coords[-20:, 0]),
                np.mean(coords[-20:, 1])
            ]
            return np.array(features, dtype=np.float32)
        except Exception as e:
            print(f"    Warning: Feature extraction error: {str(e)}")
            return np.zeros(15, dtype=np.float32)
    
    def create_sequences(self, folder_path):
        print(f"🔄 Creating sequences from {folder_path}")
        if self.tlx_data is None:
            self.load_tlx_data()
        csv_files = [f for f in os.listdir(folder_path) 
                    if f.endswith('.csv') and 'tlx' not in f.lower()]
        sequences_X = []
        sequences_y = []
        sequences_meta = []
        for filename in csv_files:
            print(f"  Processing {filename}...")
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                left_ear = np.array([self.parse_ear_string(x) for x in df['leftEAR']])
                right_ear = np.array([self.parse_ear_string(x) for x in df['rightEAR']])
                ear = df['EAR'].values.astype(np.float32)
                coords_x_list = [self.parse_coordinate_string(x) for x in df['behaviroal_AU_X']]
                coords_y_list = [self.parse_coordinate_string(x) for x in df['behaviroal_AU_Y']]
                cognitive_features = []
                for cx, cy in zip(coords_x_list, coords_y_list):
                    feat = self.extract_cognitive_features(cx, cy)
                    cognitive_features.append(feat)
                cognitive_features = np.array(cognitive_features)
                if len(left_ear) == len(right_ear) == len(ear) == len(cognitive_features):
                    combined_features = np.column_stack([
                        left_ear, right_ear, ear, cognitive_features
                    ])
                else:
                    print(f"    Warning: Feature length mismatch in {filename}")
                    continue
            except Exception as e:
                print(f"    Error processing {filename}: {str(e)}")
                continue
            condition = filename.replace('.csv', '')
            tlx_row = None
            if 'condition' in self.tlx_data.columns:
                tlx_row = self.tlx_data[self.tlx_data['condition'] == condition]
            if tlx_row is None or len(tlx_row) == 0:
                try:
                    parts = condition.split('_')
                    if len(parts) >= 4:
                        participant = parts[0].replace('P', '')
                        camera = parts[2]
                        speed = parts[4]
                        for idx, row in self.tlx_data.iterrows():
                            tlx_participant = str(row.get('Participant', row.get('participant', ''))).replace('P', '')
                            tlx_camera = str(row.get('Camera_Number', row.get('camera', row.get('Camera', ''))))
                            tlx_speed = str(row.get('Robot_speed', row.get('speed', row.get('Speed', ''))))
                            if (tlx_participant == participant and 
                                tlx_camera == camera and 
                                tlx_speed == speed):
                                tlx_row = self.tlx_data.iloc[idx:idx+1]
                                break
                except:
                    pass
            if tlx_row is None or len(tlx_row) == 0:
                print(f"    Warning: Using fallback TLX mapping for {filename}")
                tlx_row = self.tlx_data.iloc[0:1]
            tlx_values = []
            for target in self.tlx_targets:
                if target in tlx_row.columns:
                    tlx_values.append(tlx_row[target].iloc[0])
                else:
                    tlx_values.append(10.0)
            tlx_values = np.array(tlx_values)
            if len(combined_features) < self.sequence_length:
                continue
            combined_features_norm = self.feature_scaler.fit_transform(combined_features)
            step_size = int(self.sequence_length * (1 - self.overlap))
            for start_idx in range(0, len(combined_features_norm) - self.sequence_length + 1, step_size):
                sequence = combined_features_norm[start_idx:start_idx + self.sequence_length]
                target_size = self.spatial_height * self.spatial_width
                if sequence.shape[1] > target_size:
                    sequence = sequence[:, :target_size]
                elif sequence.shape[1] < target_size:
                    padding = np.zeros((self.sequence_length, target_size - sequence.shape[1]))
                    sequence = np.concatenate([sequence, padding], axis=1)
                sequence_3d = sequence.reshape(
                    self.temporal_dim, self.spatial_height, self.spatial_width, self.channels
                )
                sequences_X.append(sequence_3d)
                sequences_y.append(tlx_values)
                sequences_meta.append({
                    'filename': filename,
                    'condition': condition,
                    'start_frame': start_idx
                })
        X = np.array(sequences_X)
        y = np.array(sequences_y)
        print(f"✅ Created {len(X)} sequences: X{X.shape}, y{y.shape}")
        return X, y, sequences_meta
    
    def build_model(self):
        input_shape = (self.temporal_dim, self.spatial_height, self.spatial_width, self.channels)
        inputs = layers.Input(shape=input_shape)
        print(f"🏗️ Building model: {input_shape}")
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv3D(48, (5, 3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 1))(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Conv3D(64, (5, 3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 1))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv3D(96, (3, 2, 2), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 1, 1))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.25)(x)
        outputs = layers.Dense(5, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.0005, amsgrad=True)
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        return model
    
    def train_model(self, epochs=250):
        print("🚀 TRAINING STATE-OF-THE-ART MODEL")
        print("="*70)
        X_train, y_train, train_meta = self.create_sequences(self.train_dir)
        X_val, y_val, val_meta = self.create_sequences(self.val_dir)
        y_train_norm = self.target_scaler.fit_transform(y_train)
        y_val_norm = self.target_scaler.transform(y_val)
        print(f"Training: {X_train.shape} sequences")
        print(f"Validation: {X_val.shape} sequences")
        self.model = self.build_model()
        print(f"Model parameters: {self.model.count_params():,}")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=20, min_lr=1e-8, verbose=1),
            ModelCheckpoint(os.path.join(self.output_dir, 'model.h5'), 
                          monitor='val_loss', save_best_only=True, verbose=1)
        ]
        print(f"🎯 Starting training for up to {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train_norm,
            validation_data=(X_val, y_val_norm),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        self.model = tf.keras.models.load_model(os.path.join(self.output_dir, 'model.h5'))
    
    def evaluate_model(self):
        print("\n🏆 EVALUATING MODEL")
        print("="*50)
        X_test, y_test, test_meta = self.create_sequences(self.test_dir)
        y_test_norm = self.target_scaler.transform(y_test)
        print(f"Test: {X_test.shape} sequences")
        y_pred_norm = self.model.predict(X_test, verbose=1)
        y_pred = self.target_scaler.inverse_transform(y_pred_norm)
        results = {}
        print(f"\n📊 PERFORMANCE RESULTS:")
        print("-" * 60)
        total_r2 = 0
        total_corr = 0
        for i, subscale in enumerate(self.tlx_targets):
            y_true = y_test[:, i]
            y_pred_sub = y_pred[:, i]
            mae = mean_absolute_error(y_true, y_pred_sub)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_sub))
            r2 = r2_score(y_true, y_pred_sub)
            correlation = np.corrcoef(y_true, y_pred_sub)[0, 1] if len(np.unique(y_true)) > 1 else 0.0
            total_r2 += r2
            total_corr += correlation
            results[subscale.replace('NASA_', '')] = {
                'mae': mae, 'rmse': rmse, 'r2': r2, 'correlation': correlation
            }
            print(f"{subscale.replace('NASA_', ''):<15}: MAE={mae:6.3f}, RMSE={rmse:6.3f}, R²={r2:7.4f}, Corr={correlation:7.4f}")
        avg_r2 = total_r2 / 5
        avg_corr = total_corr / 5
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"Average R²:           {avg_r2:.4f}")
        print(f"Average Correlation:  {avg_corr:.4f}")
        print(f"Target R² > 0.8:      {'🎉 ACHIEVED' if avg_r2 > 0.8 else '⚠️ NOT ACHIEVED'}")
        results_df = pd.DataFrame(results).T
        results_df.to_csv(os.path.join(self.output_dir, 'results.csv'))
        return results, avg_r2 > 0.8, avg_r2
    
    def run_complete_pipeline(self, epochs=250):
        try:
            self.train_model(epochs=epochs)
            results, success, avg_r2 = self.evaluate_model()
            print(f"\n🎉 PIPELINE COMPLETED!")
            print("="*50)
            print(f"✅ Model trained successfully")
            print(f"✅ R² Achievement: {'SUCCESS' if success else 'PROGRESS'} ({avg_r2:.4f})")
            print(f"✅ Results saved to: {self.output_dir}/results.csv")
            print("="*50)
            return results
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    train_dir = "/workspace/Jaynab/split/train"
    val_dir = "/workspace/Jaynab/split/val"
    test_dir = "/workspace/Jaynab/split/test"
    output_dir = "/workspace/Jaynab/results"
    pipeline = StateOfArtModel(train_dir, val_dir, test_dir, output_dir)
    results = pipeline.run_complete_pipeline(epochs=250)
