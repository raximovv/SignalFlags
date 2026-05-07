# SignalFlags
SignalFlags is a real-time semaphore gesture recognition system that uses your webcam to identify hand signals and decode them into letters and words.
It trains a machine learning classifier on hand landmark data collected from MediaPipe, then runs live inference to recognize semaphore alphabet poses with confidence filtering — only outputting a letter when the model is at least 85% sure.
The core idea mirrors how gesture recognition works on embedded wearables like smartwatches or sign-language gloves: a lightweight classifier trained on spatial hand features, deployable on low-power hardware with TensorFlow Lite Micro.
Stack: MediaPipe Hands · scikit-learn · OpenCV · NumPy
