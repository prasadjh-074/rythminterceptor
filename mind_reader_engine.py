import itertools
import numpy as np


class MindReaderLogisticRegression:
    """
    Pure-NumPy Online Logistic Regression trained via Stochastic Gradient Descent (SGD)
    with L2 forgetting decay to crack human Left (0) / Right (1) keystroke patterns in real time.
    """

    def __init__(self, lr: float = 0.35, l2_decay: float = 0.015):
        self.lr = lr
        self.l2_decay = l2_decay
        self.feature_names = self._build_feature_names()
        self.n_features = len(self.feature_names)
        self.feature_index = {name: i for i, name in enumerate(self.feature_names)}
        self.reset()

    def _build_feature_names(self):
        names = []
        # 1-gram, 2-gram, 3-gram, and 4-gram sequence buckets (2 + 4 + 8 + 16 = 30 features)
        for k in (1, 2, 3, 4):
            for seq in itertools.product(["L", "R"], repeat=k):
                names.append(f"{k}gram_{''.join(seq)}")

        # 5 Behavioral & Neurological Telemetry Features
        names.extend([
            "streak_run_length",     # Captures Gambler's Fallacy (streak aversion)
            "fast_tap_last_dir",     # <220ms rapid tapping motor alternation
            "hesitation_last_dir",   # >500ms cognitive pause shift
            "win_stay_last_dir",     # Habit right after player beats AI
            "lose_shift_last_dir",   # Panic habit right after AI reads player's mind
        ])
        return names

    def reset(self):
        self.w = np.zeros(self.n_features, dtype=np.float64)
        self.b = 0.0
        self.history = []          # List of 'L' or 'R'
        self.outcomes = []         # True if AI guessed right on turn t, else False
        self.round_num = 0
        self.ai_wins = 0

    @staticmethod
    def _sigmoid(z: float) -> float:
        z_clipped = float(np.clip(z, -20.0, 20.0))
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def extract_features(self, dt_ms: float = 250.0) -> np.ndarray:
        """Builds the active feature vector x_t BEFORE the user presses the next key."""
        x = np.zeros(self.n_features, dtype=np.float64)
        h = self.history
        n = len(h)

        if n == 0:
            return x

        # 1. Activate 1-gram through 4-gram pattern buckets
        for k in (1, 2, 3, 4):
            if n >= k:
                gram_key = f"{k}gram_{''.join(h[-k:])}"
                idx = self.feature_index.get(gram_key)
                if idx is not None:
                    x[idx] = 1.0

        # 2. Streak length (negative for Left streak, positive for Right streak)
        last_move = h[-1]
        streak = 0
        for move in reversed(h):
            if move == last_move:
                streak += 1
            else:
                break
        direction_sign = 1.0 if last_move == "R" else -1.0
        x[self.feature_index["streak_run_length"]] = direction_sign * min(streak, 5) / 3.0

        # 3. Motor Rhythm Coupling (fast tap <220ms vs hesitation >500ms)
        if dt_ms < 220.0:
            x[self.feature_index["fast_tap_last_dir"]] = direction_sign
        elif dt_ms > 500.0:
            x[self.feature_index["hesitation_last_dir"]] = direction_sign

        # 4. Win-Stay / Lose-Shift psychological state
        if len(self.outcomes) > 0:
            ai_won_last = self.outcomes[-1]
            if ai_won_last:
                x[self.feature_index["lose_shift_last_dir"]] = direction_sign
            else:
                x[self.feature_index["win_stay_last_dir"]] = direction_sign

        return x

    def predict(self, dt_ms: float = 250.0):
        """
        Locks in the AI's mind-read BEFORE the player's move is revealed.
        Returns predicted move ('L' or 'R'), P(Right), log-odds z, and top feature reasons.
        """
        x = self.extract_features(dt_ms)
        contribs = self.w * x
        z = float(np.sum(contribs) + self.b)
        prob_right = self._sigmoid(z)
        pred_move = "R" if prob_right >= 0.5 else "L"
        confidence = prob_right if pred_move == "R" else (1.0 - prob_right)

        # Extract top active log-odds reasons
        active_idx = np.where(np.abs(contribs) > 0.01)[0]
        sorted_idx = active_idx[np.argsort(np.abs(contribs[active_idx]))[::-1]][:4]
        reasons = [
            {
                "feature": self.feature_names[i],
                "log_odds": round(float(contribs[i]), 3),
                "pushes_toward": "R" if contribs[i] > 0 else "L",
            }
            for i in sorted_idx
        ]

        return {
            "x": x,
            "pred_move": pred_move,
            "prob_right": round(prob_right, 4),
            "confidence": round(confidence, 4),
            "log_odds": round(z, 3),
            "reasons": reasons,
        }

    def step(self, actual_move: str, dt_ms: float = 250.0):
        """
        1. Predicts the user's move using current weights.
        2. Compares against actual_move ('L' or 'R').
        3. Updates weights immediately via Online SGD with L2 decay.
        """
        pred_info = self.predict(dt_ms)
        x = pred_info["x"]
        prob_right = pred_info["prob_right"]
        pred_move = pred_info["pred_move"]

        y = 1.0 if actual_move == "R" else 0.0
        ai_correct = (pred_move == actual_move)

        # Online SGD Update: w_{t+1} = (1 - lr*decay)*w_t - lr * (p - y) * x_t
        error = prob_right - y
        self.w = (1.0 - self.lr * self.l2_decay) * self.w - self.lr * error * x
        self.b = (1.0 - self.lr * self.l2_decay) * self.b - (self.lr * 0.25) * error

        # Update state history
        self.round_num += 1
        if ai_correct:
            self.ai_wins += 1
        self.history.append(actual_move)
        self.outcomes.append(ai_correct)

        return {
            "round": self.round_num,
            "actual_move": actual_move,
            "ai_prediction": pred_move,
            "ai_correct": ai_correct,
            "prob_right": prob_right,
            "confidence": pred_info["confidence"],
            "log_odds": pred_info["log_odds"],
            "error": round(float(error), 3),
            "ai_win_rate": round(self.ai_wins / self.round_num, 4),
            "reasons": pred_info["reasons"],
        }


if __name__ == "__main__":
    # Quick diagnostic: Simulate a human player with realistic habits over 100 keystrokes
    np.random.seed(42)
    engine = MindReaderLogisticRegression(lr=0.35, l2_decay=0.015)

    print("🧠 Testing Mind-Reader Online SGD Engine against a simulated human (100 moves)...\n")
    sim_moves = ["L", "R"]
    for t in range(100):
        # Simulated human bias: hates 3-in-a-row streaks & prefers 'R' after 'L-R-L'
        if len(sim_moves) >= 3 and sim_moves[-3:] == ["L", "R", "L"]:
            p_r = 0.82
        elif len(sim_moves) >= 2 and sim_moves[-1] == sim_moves[-2]:
            p_r = 0.80 if sim_moves[-1] == "L" else 0.20
        else:
            p_r = 0.58 if sim_moves[-1] == "L" else 0.42

        move = "R" if np.random.rand() < p_r else "L"
        sim_moves.append(move)
        res = engine.step(move, dt_ms=190.0)

        if res["round"] in (10, 25, 50, 75, 100):
            print(
                f"Round {res['round']:3d} | AI Win Rate: {res['ai_win_rate']*100:5.1f}% "
                f"| Last Guess: {res['ai_prediction']} (Conf: {res['confidence']*100:4.1f}%) "
                f"| Top Signal: {res['reasons'][0]['feature'] if res['reasons'] else 'Warming up'}"
            )

    print("\n✅ Phase 1 Engine Ready! Top 5 Learned Weights (Log-Odds):")
    top_idx = np.argsort(np.abs(engine.w))[::-1][:5]
    for idx in top_idx:
        print(f"  {engine.feature_names[idx]:20s}: {engine.w[idx]:+.3f}")