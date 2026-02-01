import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (balanced_accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score)
import tensorflow as tf

def check_tensorflow_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow is using GPU: {gpus[0].name}")
        # Optional: Set memory growth to prevent TF from hugging all VRAM
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("❌ TensorFlow is running on CPU! (Check your drivers/Colab runtime)")

# Assumes you have the EEGNet model file we created earlier
from classification.models.eeg_net.eeg_net import EEGNet

def train_eeg_net(X_train, y_train, 
                 X_val, y_val,            # MANDATORY: Used to find "Best Weights"
                 X_test=None, y_test=None,# OPTIONAL: Only used for final paper/report results
                 lr=1e-3, 
                 dropoutRate=0.5, 
                 batch_size=64, 
                 epochs=100):
    
    # --- 1. Setup & Dimensions ---
    nb_classes = 2  # Hardcoded as requested
    Chans = X_train.shape[1]
    Samples = X_train.shape[2]
    
    # Dynamic Kernel Scaling (Standard is 0.5s: 64 for 128Hz -> 250 for 500Hz)
    kernLength = 250 if Samples > 400 else 64 
    print(f"Training EEGNet | Input: {X_train.shape} | KernLength: {kernLength}")

    check_tensorflow_gpu()


    # --- 2. Build Model ---
    model = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, 
                   dropoutRate=dropoutRate, kernLength=kernLength)

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  metrics=['accuracy'])

    # --- 3. Callbacks (The Magic Happens Here) ---
    callbacks = [
        # restore_best_weights=True: When training ends, Keras overwrites the model 
        # with the weights that had the LOWEST loss on X_val.
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
    ]

    # --- 4. Train ---
    print("\n[Status] Training (Monitoring X_val for best weights)...")
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), # Used strictly for picking best weights
              batch_size=batch_size, 
              epochs=epochs, 
              callbacks=callbacks,
              verbose=1)

    # --- 5. Logic: Optimization vs Final Test ---
    # At this point, 'model' ALREADY contains the best weights from the validation step.
    
    if X_test is not None:
        print("\n[Status] Best weights restored. Evaluating on TEST set...")
        eval_X, eval_y = X_test, y_test
    else:
        print("\n[Status] Best weights restored. Evaluating on VALIDATION set (Optimization mode)...")
        eval_X, eval_y = X_val, y_val

    # --- 6. Calculate Metrics ---
    probs = model.predict(eval_X, verbose=0)
    preds = np.argmax(probs, axis=1) # Convert probabilities to 0 or 1
    
    bal_acc = balanced_accuracy_score(eval_y, preds)
    conf_matrix = confusion_matrix(eval_y, preds)
    report = classification_report(eval_y, preds, target_names=['Non-Target', 'Target'], zero_division=0)
    
    # Binary AUROC: Use probability of the "Target" class (Index 1)
    try:
        auroc = roc_auc_score(eval_y, probs[:, 1])
    except ValueError:
        auroc = 0.5 

    results = {
        'score': bal_acc, 
        'confusion_matrix': conf_matrix,
        'report': report,
        'auroc': auroc
    }

    print("\n--- Final Results (Best Epoch) ---")
    print(results['report'])
    print(f"Balanced Acc: {bal_acc:.4f} | AUROC: {auroc:.4f}")

    K.clear_session()
    return results