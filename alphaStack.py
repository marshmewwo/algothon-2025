import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class ConfirmedBreakoutEnsembleStrategyV3:
    def __init__(self):
        # --- Competition-optimized configuration ---
        self.fast_ma_window = 20
        self.slow_ma_window = 50
        self.signal_confirmation_period = 3
        self.min_training_days = 100
        self.retrain_interval = 25
        self.prediction_horizon = 5
        self.ml_confidence_factor = 2.0
        self.volatility_window = 50
        self.volatility_smoothing_alpha = 0.13
        self.base_min_signal_for_entry = 0.25
        self.breakout_window = 40
        self.adjustment_threshold_breakout = 0.10
        self.blend_weight = 0.25
        self.target_dollar_per_inst = 800.0
        self.volatility_kill_switch_percentile = 85
        self.max_portfolio_dollar_vol = 600.0
        self.position_smoothing_factor = 0.85
        self.max_total_dollar_exposure = 275_000

        self.base_adjustment_threshold_normal = 0.65

        # Internal State
        self.model_pipeline = None
        self.last_train_day = -1
        self.prvPos = None
        self.smoothed_instrument_vol = None

    def _extract_features(self, prices: np.ndarray) -> np.ndarray:
        n = len(prices)
        if n < self.slow_ma_window: return np.zeros(4)
        price_then = prices[-10]
        f1_momentum = (prices[-1] / price_then) - 1 if price_then > 1e-6 else 0
        returns = np.diff(np.log(prices[-21:]))
        f2_volatility = np.std(returns) if len(returns) > 1 else 0
        deltas = np.diff(prices[-15:])
        gains = np.sum(deltas[deltas > 0]) / 14
        losses = np.sum(-deltas[deltas < 0]) / 14
        rs = gains / (losses + 1e-9)
        f3_rsi = 100 - (100 / (1 + rs))
        sma_fast = np.mean(prices[-self.fast_ma_window:])
        f4_price_vs_sma = (prices[-1] / sma_fast) - 1 if sma_fast > 1e-6 else 0
        return np.array([f1_momentum, f2_volatility, f3_rsi, f4_price_vs_sma])

    def _train_model(self, prcSoFar: np.ndarray):
        nInst, nt = prcSoFar.shape
        X_train, y_train = [], []
        start_day, end_day = self.slow_ma_window, nt - self.prediction_horizon
        for day in range(start_day, end_day):
            for i in range(nInst):
                price_hist = prcSoFar[i, :day + 1]
                features = self._extract_features(price_hist)
                current_price, future_price = price_hist[-1], prcSoFar[i, day + self.prediction_horizon]
                if current_price > 1e-6 and future_price > 0:
                    y_train.append((future_price / current_price) - 1)
                    X_train.append(features)
        if len(y_train) < 500: return
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', HistGradientBoostingRegressor(random_state=42, max_iter=100, max_depth=5))
        ])
        try:
            self.model_pipeline.fit(np.array(X_train), np.array(y_train))
            self.last_train_day = nt
        except Exception:
            self.model_pipeline = None

    def _get_adjustment_threshold_normal(self, cs_vol):
        return self.base_adjustment_threshold_normal * (1 + 0.10 * cs_vol)

    def getMyPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        nInst, nt = prcSoFar.shape
        if self.prvPos is None: self.prvPos = np.zeros(nInst)

        # --- WARMUP & RAMP-UP ADDITION ---
        required_data = max(self.slow_ma_window + self.signal_confirmation_period,
                            self.breakout_window + 2,
                            self.min_training_days)
        if nt < required_data:
            return np.zeros(nInst, dtype=int)

        # Ramp up risk for first warmup_days after ML is ready
        warmup_days = 50
        if nt < self.min_training_days + warmup_days:
            ramp = min(1.0, (nt - self.min_training_days) / warmup_days)
        else:
            ramp = 1.0

        # --- Signal Generation ---
        fast_mas = np.array([np.mean(prcSoFar[:, nt - self.fast_ma_window - i:nt - i], axis=1) for i in
                             range(self.signal_confirmation_period, 0, -1)]).T
        slow_mas = np.array([np.mean(prcSoFar[:, nt - self.slow_ma_window - i:nt - i], axis=1) for i in
                             range(self.signal_confirmation_period, 0, -1)]).T
        confirmed_ma_signal = np.where(np.all(np.sign(fast_mas - slow_mas) == 1, axis=1), 1,
                                       np.where(np.all(np.sign(fast_mas - slow_mas) == -1, axis=1), -1, 0))

        needs_training = self.model_pipeline is None or (nt - self.last_train_day) >= self.retrain_interval
        if needs_training and nt >= self.min_training_days: self._train_model(prcSoFar)

        raw_confidence = np.full(nInst, 0.5)
        if self.model_pipeline is not None:
            X_today = np.array([self._extract_features(prcSoFar[i, :]) for i in range(nInst)])
            raw_confidence = 1 / (1 + np.exp(
                -self.ml_confidence_factor * (self.model_pipeline.predict(X_today) * confirmed_ma_signal)))

        momentum_scores = (prcSoFar[:, -1] / prcSoFar[:, -21]) - 1
        momentum_scores = (momentum_scores - np.mean(momentum_scores)) / (np.std(momentum_scores) + 1e-9)

        # Fallback to momentum-only signal before ML is trained
        if self.model_pipeline is None:
            final_signal = momentum_scores.copy()
        else:
            ml_signal = confirmed_ma_signal * raw_confidence
            agreed = np.sign(ml_signal) == np.sign(momentum_scores)
            final_signal = np.where(agreed, (1 - self.blend_weight) * ml_signal + self.blend_weight * momentum_scores,
                                    ml_signal)

        # Micro-normalize the signal (z-score for cross-sectional risk stability)
        final_signal = (final_signal - np.mean(final_signal)) / (np.std(final_signal) + 1e-9)

        cs_vol = np.std(final_signal)
        min_signal_for_entry = self.base_min_signal_for_entry * (1 + 0.20 * cs_vol)
        final_signal[np.abs(final_signal) < min_signal_for_entry] = 0

        # --- Sizing and Risk Management ---
        upper_channels = np.max(prcSoFar[:, -self.breakout_window - 1:-1], axis=1)
        lower_channels = np.min(prcSoFar[:, -self.breakout_window - 1:-1], axis=1)
        current_prices = prcSoFar[:, -1]
        breakout_signal = np.zeros(nInst)
        breakout_signal[current_prices > upper_channels] = 1.0
        breakout_signal[current_prices < lower_channels] = -1.0

        recent_prices = prcSoFar[:, -self.volatility_window - 1:]
        returns = np.nan_to_num(
            (recent_prices[:, 1:] - recent_prices[:, :-1]) / np.maximum(recent_prices[:, :-1], 1e-9))
        raw_instrument_vol = np.std(returns, axis=1)
        raw_instrument_vol[raw_instrument_vol == 0] = 1e-6
        if self.smoothed_instrument_vol is None:
            self.smoothed_instrument_vol = raw_instrument_vol
        else:
            self.smoothed_instrument_vol = (self.volatility_smoothing_alpha * raw_instrument_vol + (
                        1 - self.volatility_smoothing_alpha) * self.smoothed_instrument_vol)

        target_dollar_allocation = final_signal / self.smoothed_instrument_vol
        scalar = self.target_dollar_per_inst / (np.mean(np.abs(target_dollar_allocation)) + 1e-9)
        scaled_dollar_allocation = target_dollar_allocation * scalar

        # --- RAMP-UP: Multiply position size by ramp ---
        scaled_dollar_allocation *= ramp

        total_dollar = np.sum(np.abs(scaled_dollar_allocation))
        if total_dollar > self.max_total_dollar_exposure:
            scaled_dollar_allocation *= self.max_total_dollar_exposure / (total_dollar + 1e-9)

        if np.sum(np.abs(scaled_dollar_allocation)) > 1e-9:
            cov_matrix = np.cov(returns)
            portfolio_dollar_vol = np.sqrt(max(0, scaled_dollar_allocation @ cov_matrix @ scaled_dollar_allocation))
            if portfolio_dollar_vol > self.max_portfolio_dollar_vol:
                scaled_dollar_allocation *= self.max_portfolio_dollar_vol / portfolio_dollar_vol

        target_positions = np.nan_to_num(scaled_dollar_allocation / current_prices)

        vol_cut = np.percentile(self.smoothed_instrument_vol, self.volatility_kill_switch_percentile)
        target_positions[self.smoothed_instrument_vol > vol_cut] = 0

        adjustment_threshold_normal = self._get_adjustment_threshold_normal(cs_vol)
        is_confirmed_breakout = (final_signal * breakout_signal) > 0
        adjustment_threshold = np.where(is_confirmed_breakout, self.adjustment_threshold_breakout,
                                        adjustment_threshold_normal)

        is_entry = (np.abs(self.prvPos) < 1e-6) & (np.abs(target_positions) > 1e-6)
        is_exit = (np.abs(self.prvPos) > 1e-6) & (np.abs(target_positions) < 1e-6)
        is_adjustment = (np.abs(self.prvPos) > 1e-6) & (np.abs(target_positions) > 1e-6)
        adjustment_change_pct = np.abs(target_positions - self.prvPos) / (np.abs(self.prvPos) + 1e-6)
        is_significant_adjustment = is_adjustment & (adjustment_change_pct > adjustment_threshold)
        trade_mask = is_entry | is_exit | is_significant_adjustment

        alpha = 1 - self.position_smoothing_factor
        smoothed_target = (alpha * target_positions) + ((1 - alpha) * self.prvPos)

        final_positions = np.where(trade_mask, smoothed_target, self.prvPos)
        self.prvPos = final_positions

        return final_positions.astype(int)


# --- Global Instance ---
strategy_instance = ConfirmedBreakoutEnsembleStrategyV3()


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    return strategy_instance.getMyPosition(prcSoFar)
