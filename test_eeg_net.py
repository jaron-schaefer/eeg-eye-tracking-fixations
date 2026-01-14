from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from models.eeg_net import EEGNet
from data.data_loader import load_and_prepare_eegnet_data
from data.data_configs import DEFAULT_EEGNET_CHANNELS, SUBJECTS
import numpy as np
import os


def eegnet_test(subject):
    """Test EEGNet with a single subject's data"""
    print(f"Testing EEGNet with subject {subject}")
    
    # Load data using our new functions
    X_train, X_val, X_test, y_train, y_val, y_test, selected_channels = load_and_prepare_eegnet_data(
        subject=subject,
        desired_channels=DEFAULT_EEGNET_CHANNELS,
        test_size=0.3,
        random_state=42
    )
    
    # Get dimensions from loaded data
    chans = X_train.shape[1]
    samples = X_train.shape[2]
    
    print(f"Data shape: {X_train.shape}, Channels: {len(selected_channels)}")
    
    # Create model with actual data dimensions
    model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                   dropoutRate=0.25, kernLength=32, F1=16, D=2, F2=32,
                   dropoutType='Dropout')
    
    optimizer = SGD(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=f'./baseline_{subject}.h5', verbose=1)

    fittedModel = model.fit(X_train, y_train, batch_size=32, epochs=10,
                            verbose=1, validation_data=(X_val, y_val), callbacks=[checkpointer])

    # Only load best weights if they were saved
    checkpoint_path = f'./baseline_{subject}.h5'
    if os.path.exists(checkpoint_path):
        print("Loading best weights from checkpoint")
        model.load_weights(checkpoint_path)
    else:
        print("No checkpoint saved - using final epoch weights")

    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == y_test.argmax(axis=-1))
    
    print(f"Subject {subject} - Classification accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    # Test with available subjects that have training data
    test_subjects = ["P01"]  # Start with P01, add more as data becomes available
    
    os.makedirs('models', exist_ok=True)
    
    acc_results = []
    
    for subject in test_subjects:
        try:
            acc_temp = eegnet_test(subject)
            acc_results.append(f"{subject}: {acc_temp:.4f}")
        except FileNotFoundError as e:
            print(f"No training data found for {subject}: {e}")
        except Exception as e:
            print(f"Error testing {subject}: {e}")
    
    # Save results
    with open('./EEGNet_acc.txt', 'w') as file:
        file.write('\n'.join(acc_results))
    
    print(f"\nResults saved to EEGNet_acc.txt:")
    for result in acc_results:
        print(result)
