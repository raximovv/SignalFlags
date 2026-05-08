import cv2
import numpy as np
import sys
import os
import time
import collections

try:
    import mediapipe as mp
except ImportError:
    print("pip install mediapipe"); sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("pip install scikit-learn"); sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("pip install pandas"); sys.exit(1)


def find_dataset():
    candidates = [
        "semaphore_landmarks.csv",
        "assets/semaphore_landmarks.csv",
        "../../assets/semaphore_landmarks.csv",
        "../../../assets/semaphore_landmarks.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_dataset(path):
    df = pd.read_csv(path)
    labels = df["letter"].values
    features = df.drop("letter", axis=1).values
    return features, labels


def normalize_landmarks(landmarks_flat):
    pts = landmarks_flat.reshape(21, 3)
    wrist = pts[0]
    pts = pts - wrist
    hand_size = np.linalg.norm(pts[9])
    if hand_size > 1e-6:
        pts = pts / hand_size
    return pts.flatten()


CLASSIFIER_TYPE = "random_forest"


def build_classifier():
    if CLASSIFIER_TYPE == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    else:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )


def train(features, labels):
    norm_features = np.array([normalize_landmarks(f) for f in features])
    X_train, X_test, y_train, y_test = train_test_split(
        norm_features, labels,
        test_size=0.2, random_state=42, stratify=labels
    )
    clf = build_classifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"\n Classifier: {CLASSIFIER_TYPE}")
    print(f" Training samples: {len(X_train)}")
    print(f" Test samples: {len(X_test)}")
    print(f" Test accuracy: {acc*100:.1f}%")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(labels)))
    per_class = cm.diagonal() / cm.sum(axis=1)
    sorted_letters = sorted(set(labels))
    worst = sorted(zip(sorted_letters, per_class), key=lambda x: x[1])[:5]
    print(f"\n Lowest accuracy letters:")
    for letter, acc_l in worst:
        print(f"   {letter}: {acc_l*100:.0f}%")
    return clf


CONFIDENCE_THRESHOLD = 0.85

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)


def extract_landmarks(hand_landmarks):
    pts = []
    for lm in hand_landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.z])
    return np.array(pts)


def predict_letter(clf, landmarks_raw):
    norm = normalize_landmarks(landmarks_raw)
    proba = clf.predict_proba([norm])[0]
    confidence = np.max(proba)
    letter = clf.classes_[np.argmax(proba)]
    return letter, confidence


class WordDecoder:
    def __init__(self, commit_frames=10, gap_frames=15):
        self.commit_frames = commit_frames
        self.gap_frames = gap_frames
        self.current_letter = None
        self.frame_count = 0
        self.word = ""
        self.committed_letters = []
        self.gap_counter = 0
        self.last_committed = None

    def update(self, letter, confidence):
        if letter is None or confidence < CONFIDENCE_THRESHOLD:
            self.gap_counter += 1
            if self.gap_counter > self.gap_frames:
                self.current_letter = None
                self.frame_count = 0
            return None
        self.gap_counter = 0
        if letter == self.current_letter:
            self.frame_count += 1
            if self.frame_count == self.commit_frames:
                if letter != self.last_committed:
                    self.word += letter
                    self.last_committed = letter
                    return letter
        else:
            self.current_letter = letter
            self.frame_count = 1
        return None

    def clear_word(self):
        self.word = ""
        self.last_committed = None


def main():
    print("\n" + "=" * 55)
    print(" SignalFlags — Day 27")
    print("=" * 55)

    dataset_path = find_dataset()
    if dataset_path is None:
        print("\nERROR: semaphore_landmarks.csv not found.")
        print("It should be in assets/ folder from your Day 0 repo.")
        print("Run: git pull in your buildcored-orcas folder")
        sys.exit(1)

    print(f"\n Dataset: {dataset_path}")
    features, labels = load_dataset(dataset_path)
    print(f" {len(features)} samples | {len(set(labels))} classes | {features.shape[1]} features")

    print("\nTraining classifier...")
    clf = train(features, labels)
    print("\n Classifier ready")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No webcam found.")
        sys.exit(1)

    decoder = WordDecoder()
    print("\n Webcam active. Show semaphore hand poses.")
    print("Press 'c' to clear word | 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        letter = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display, hand_lm, mp_hands.HAND_CONNECTIONS)
            landmarks_raw = extract_landmarks(hand_lm)
            letter, confidence = predict_letter(clf, landmarks_raw)

        committed = decoder.update(letter, confidence)
        if committed:
            print(f" Committed: {committed} -> Word: '{decoder.word}'")

        if decoder.current_letter and confidence >= CONFIDENCE_THRESHOLD:
            progress = decoder.frame_count / decoder.commit_frames
            bar_w = int(progress * 200)
            cv2.rectangle(display, (10, h - 50), (210, h - 30), (50, 50, 50), -1)
            cv2.rectangle(display, (10, h - 50), (10 + bar_w, h - 30), (0, 255, 0), -1)
            cv2.putText(display, f"Holding: {decoder.current_letter}",
                        (220, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if letter and confidence >= CONFIDENCE_THRESHOLD:
            cv2.putText(display, f"{letter}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
            cv2.putText(display, f"{confidence*100:.0f}%",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        elif letter:
            cv2.putText(display, f"{letter} ({confidence*100:.0f}%)",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
            cv2.putText(display, "LOW CONFIDENCE",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        else:
            cv2.putText(display, "Show hand pose",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        word_display = decoder.word if decoder.word else "_"
        cv2.putText(display, f"Word: {word_display}",
                    (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2)
        cv2.putText(display, "c=clear  q=quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("SignalFlags - Day 27", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            decoder.clear_word()
            print(" Word cleared")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal word: '{decoder.word}'")
    print("SignalFlags ended.")


if __name__ == "__main__":
    main()