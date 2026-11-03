"""学習条件の修正 (2026-07-02) に対する回帰テスト。

対象:
  F1. prediction_context.build_full_field_context — サーブ時Elo縮退の修正
  F2. Label Smoothing × LGBM objective=binary の退化防止 (cross_entropy 切替)
  F3. Focal Loss の平滑化ラベル対応 (y_true > 0.5 判定)
  F4. Adversarial Validation の再実装 (単一クラス学習バグの修正)
  F5. L2/L4 メタ入力スケールのフラグゲート (l2_inputs_calibrated)
  F6. is_outlier から当該レース走破タイムを除外
  F7. オッズ派生特徴量のデフォルト除外ポリシー
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from model_trainer import RacePredictionModel, _focal_loss_lgbm
from data_processor import RaceDataProcessor, FeatureEngineer
from features_advanced import AdvancedFeatureEngineer
from prediction_context import build_full_field_context, featurize_with_full_context


class _FakeDtrain:
    def __init__(self, labels):
        self._labels = np.asarray(labels, dtype=float)

    def get_label(self):
        return self._labels


# ----------------------------------------------------------------------
# F3: Focal Loss × Label Smoothing
# ----------------------------------------------------------------------
class TestFocalLossSmoothedLabels:
    def test_positive_samples_get_positive_class_weighting(self):
        """平滑化ラベル (0.975) の正例が負例扱いされないこと。

        旧実装は y_true == 1 判定のため、平滑化後は全サンプルが
        負例扱い (at=1-alpha, pt=1-p) になり、確信度の低い正例
        (p が小さい) の勾配がほぼ消えていた。
        """
        obj, _ = _focal_loss_lgbm(gamma=2.0, alpha=0.25)
        # 正例(平滑化済み 0.975) を p=0.1 (自信なし) と予測しているケース
        y = [0.975]
        raw = [np.log(0.1 / 0.9)]  # sigmoid^-1(0.1)
        grad, hess = obj(np.array(raw), _FakeDtrain(y))
        # 正しい focal 重み: at=0.25, pt=p=0.1 → w=0.25*0.9^2=0.2025
        # 旧バグ: at=0.75, pt=1-p=0.9 → w=0.75*0.01=0.0075 (勾配ほぼ消滅)
        assert abs(grad[0]) > 0.1, (
            f"低確信の正例の勾配が消えている (grad={grad[0]:.5f}) — "
            "focal重みが負例扱いになっている可能性"
        )
        assert grad[0] < 0, "正例なのに勾配が確率を下げる方向"

    def test_hard_labels_unchanged(self):
        """従来の 0/1 ラベルでは挙動が変わらないこと (y>0.5 と y==1 は等価)。"""
        obj, _ = _focal_loss_lgbm(gamma=2.0, alpha=0.25)
        y = [1.0, 0.0]
        raw = [0.0, 0.0]  # p=0.5
        grad, hess = obj(np.array(raw), _FakeDtrain(y))
        # p=0.5: 正例 w=0.25*0.25, grad=w*(0.5-1)<0 / 負例 w=0.75*0.25, grad>0
        assert grad[0] < 0 and grad[1] > 0
        assert hess[0] > 0 and hess[1] > 0

    def test_eval_penalizes_confident_wrong_positive(self):
        """評価関数: 自信満々に外した正例の損失が大きいこと。"""
        _, feval = _focal_loss_lgbm(gamma=2.0, alpha=0.25)
        wrong = _FakeDtrain([0.975])
        raw_wrong = np.array([np.log(0.02 / 0.98)])  # p=0.02 (正例をほぼ否定)
        _, loss_wrong, _ = feval(raw_wrong, wrong)
        raw_right = np.array([np.log(0.98 / 0.02)])  # p=0.98
        _, loss_right, _ = feval(raw_right, wrong)
        assert loss_wrong > loss_right * 10


# ----------------------------------------------------------------------
# F2: Label Smoothing × LightGBM binary の退化防止
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def separable_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(3000, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"].values + rng.normal(scale=0.5, size=3000) > 0).astype(float)
    return X, y


class TestSmoothedBinaryTraining:
    def test_smoothed_labels_do_not_degenerate(self, separable_data):
        """平滑化ラベル (0.025/0.975) で学習した LGBM が定数予測にならないこと。

        v7.2 では objective=binary に平滑化ラベルを渡した結果、全サンプルが
        正例扱いになり top3 用 LGBM 5本が全て定数1.0を返していた (実測確認済)。
        """
        from sklearn.metrics import roc_auc_score
        X, y = separable_data
        alpha = 0.05
        y_smooth = y * (1 - alpha) + alpha * 0.5

        model = RacePredictionModel()
        m = model._train_lgbm(
            X.iloc[:2400], y_smooth[:2400], X.iloc[2400:], y_smooth[2400:], task="binary"
        )
        pred = m.predict(X.iloc[2400:])
        assert np.std(pred) > 0.01, f"予測が定数に退化 (std={np.std(pred):.6f})"
        assert pred.min() < 0.5 < pred.max(), "予測が片側に張り付いている"
        auc = roc_auc_score(y[2400:], pred)
        assert auc > 0.75, f"分離可能な合成データでAUC {auc:.3f} — 学習が機能していない"

    def test_hard_labels_still_use_binary_objective(self, separable_data):
        """0/1 ラベルでは従来通り binary objective で学習されること。"""
        from sklearn.metrics import roc_auc_score
        X, y = separable_data
        model = RacePredictionModel()
        m = model._train_lgbm(
            X.iloc[:2400], y[:2400], X.iloc[2400:], y[2400:], task="binary"
        )
        pred = m.predict(X.iloc[2400:])
        auc = roc_auc_score(y[2400:], pred)
        assert auc > 0.75
        assert 0.0 <= pred.min() and pred.max() <= 1.0


# ----------------------------------------------------------------------
# F4: Adversarial Validation
# ----------------------------------------------------------------------
class TestAdversarialValidation:
    def _make_model(self):
        return RacePredictionModel()

    def test_no_shift_gives_auc_near_half(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(2000, 5)), columns=[f"f{i}" for i in range(5)])
        model = self._make_model()
        model._run_adversarial_validation(X, pd.DataFrame())
        auc = model._adversarial_auc
        assert auc is not None
        assert 0.35 < auc < 0.65, f"同一分布でAUC {auc:.3f} (0.5近辺のはず)"

    def test_shift_is_detected(self):
        """旧実装は常に0.5を返した (単一クラス学習 → ValueError → 0.5)。
        分布シフトを実際に検出できることを確認する。"""
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.normal(size=(2000, 5)), columns=[f"f{i}" for i in range(5)])
        # 直近20%の f0 を +3σ シフト
        X.iloc[1600:, 0] += 3.0
        model = self._make_model()
        model._run_adversarial_validation(X, pd.DataFrame())
        auc = model._adversarial_auc
        assert auc is not None
        assert auc > 0.8, f"明確なシフトを検出できていない (AUC {auc:.3f})"

    def test_calendar_columns_are_excluded(self):
        """year 等のカレンダー列だけが期間と相関するケースでは
        警告を出さない (自明な分離を実質ドリフトと誤認しない)。"""
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.normal(size=(2000, 4)), columns=[f"f{i}" for i in range(4)])
        X["year"] = [2020] * 1600 + [2021] * 400  # 完全にラベルと一致
        model = self._make_model()
        model._run_adversarial_validation(X, pd.DataFrame())
        auc = model._adversarial_auc
        assert auc is not None
        assert auc < 0.65, f"カレンダー列が除外されていない (AUC {auc:.3f})"


# ----------------------------------------------------------------------
# F1: full-field context
# ----------------------------------------------------------------------
class TestBuildFullFieldContext:
    def _db(self):
        return pd.DataFrame({
            "race_id": ["r1", "r1", "r1", "r2", "r2", "r3", "r3"],
            "horse_id": ["A", "B", "C", "A", "D", "E", "F"],
            "finish_position": [1, 2, 3, 2, 1, 1, 2],
        })

    def test_includes_co_runners_of_past_races(self):
        """対象馬の過去レースについて全出走馬の行が含まれること。

        従来 (対象馬の行のみ) では過去レースが1頭立て扱いになり、
        Elo更新が発火せず horse_elo が全馬1500に縮退していた。
        """
        db = self._db()
        ctx = build_full_field_context(db, ["A"])
        # A が出た r1, r2 の全行 (B, C, D を含む) が入る
        assert set(ctx["race_id"]) == {"r1", "r2"}
        assert set(ctx["horse_id"]) == {"A", "B", "C", "D"}
        # A と無関係な r3 は含まれない
        assert "E" not in set(ctx["horse_id"])

    def test_fallback_to_own_rows_when_too_large(self):
        db = self._db()
        ctx = build_full_field_context(db, ["A"], max_rows=3)
        # フルフィールド5行 > 3 → 対象馬の行のみ (2行)
        assert set(ctx["horse_id"]) == {"A"}
        assert len(ctx) == 2

    def test_empty_horse_ids(self):
        assert build_full_field_context(self._db(), []) is None

    def test_elo_gets_variance_with_full_field_context(self):
        """full-field context で Elo が実際に分化することの統合テスト。"""
        rng = np.random.default_rng(7)
        rows = []
        horses = [f"H{i}" for i in range(8)]
        strength = {h: i for i, h in enumerate(horses)}  # H7 が最強
        for r in range(12):
            date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=7 * r)
            field = list(rng.choice(horses, size=5, replace=False))
            order = sorted(field, key=lambda h: -strength[h] + rng.normal(scale=1.0))
            for pos, h in enumerate(order, start=1):
                rows.append({
                    "race_id": f"R{r:03d}", "race_date": date.strftime("%Y-%m-%d"),
                    "race_name": "t", "course_type": "芝", "distance": 1600,
                    "weather": "晴", "track_condition": "良",
                    "horse_id": h, "horse_name": h, "age": 4, "sex": "牡",
                    "weight": 55.0, "horse_weight": 480, "weight_diff": 0,
                    "jockey_id": f"J{pos}", "jockey_name": f"J{pos}",
                    "trainer_id": f"T{pos}", "trainer_name": f"T{pos}",
                    "frame_number": pos, "horse_number": pos,
                    "odds": 5.0, "popularity": pos,
                    "finish_position": pos, "finish_time": "1:35.0",
                    "prize_money": 100 if pos == 1 else 0,
                    "passing_order": "-".join([str(pos)] * 4), "last_3f": 34.0,
                })
        db = pd.DataFrame(rows)
        target_horses = horses[:4]
        ctx = build_full_field_context(db, target_horses)

        proc = RaceDataProcessor()
        proc.raw_data = ctx
        processed = proc.preprocess()
        featured = FeatureEngineer().create_features(processed)
        assert "horse_elo" in featured.columns
        # 複数頭が同居するレースが含まれるため Elo が分化する
        assert featured["horse_elo"].nunique() > 1, (
            "full-field context でも Elo が縮退している"
        )

    def test_featurize_with_full_context_for_upcoming_race(self):
        """出馬表 (DBに無い未来レース) が本パイプラインで特徴量化されること。

        upcoming/precache 経路は従来 enrich_with_history の粗い近似のみで
        model.predict に直行しており、大半の特徴量が0埋めだった。
        """
        rng = np.random.default_rng(21)
        rows = []
        horses = [f"H{i}" for i in range(8)]
        for r in range(10):
            date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=7 * r)
            field = list(rng.choice(horses, size=5, replace=False))
            for pos, h in enumerate(field, start=1):
                rows.append({
                    "race_id": f"R{r:03d}", "race_date": date.strftime("%Y-%m-%d"),
                    "race_name": "t", "course_type": "芝", "distance": 1600,
                    "weather": "晴", "track_condition": "良",
                    "horse_id": h, "horse_name": h, "age": 4, "sex": "牡",
                    "weight": 55.0, "horse_weight": 480, "weight_diff": 0,
                    "jockey_id": f"J{pos}", "jockey_name": f"J{pos}",
                    "trainer_id": f"T{pos}", "trainer_name": f"T{pos}",
                    "frame_number": pos, "horse_number": pos,
                    "odds": 5.0, "popularity": pos,
                    "finish_position": pos, "finish_time": "1:35.0",
                    "prize_money": 100 if pos == 1 else 0,
                    "passing_order": "-".join([str(pos)] * 4), "last_3f": 34.0,
                })
        db = pd.DataFrame(rows)

        # 未来レースの出馬表 (結果カラムなし = 実際のスクレイプと同じ)
        target_horses = horses[:4]
        shutuba = pd.DataFrame({
            "race_id": ["R999"] * 4,
            "race_date": ["2025-06-01"] * 4,
            "race_name": ["未来S"] * 4,
            "course_type": ["芝"] * 4, "distance": [1600] * 4,
            "weather": ["晴"] * 4, "track_condition": ["良"] * 4,
            "horse_id": target_horses, "horse_name": target_horses,
            "age": [4] * 4, "sex": ["牡"] * 4,
            "weight": [55.0] * 4, "horse_weight": [480] * 4, "weight_diff": [0] * 4,
            "jockey_id": ["J1", "J2", "J3", "J4"],
            "trainer_id": ["T1", "T2", "T3", "T4"],
            "frame_number": [1, 2, 3, 4], "horse_number": [1, 2, 3, 4],
            "odds": [2.0, 4.0, 8.0, 16.0], "popularity": [1, 2, 3, 4],
        })

        featured = featurize_with_full_context(shutuba, db)
        assert featured is not None and len(featured) == 4
        # 履歴系特徴量が実際に計算されている (0埋めでない)
        assert featured["horse_elo"].nunique() > 1, "Eloが計算されていない"
        assert "horse_avg_pos_3" in featured.columns
        assert featured["horse_avg_pos_3"].notna().all()
        # 対象レースの結果カラムはマスクされている
        assert featured["finish_position"].isna().all()

    def test_featurize_returns_none_on_bad_input(self):
        db = self._db()
        assert featurize_with_full_context(pd.DataFrame(), db) is None
        assert featurize_with_full_context(None, db) is None

    def test_combined_elo_survives_missing_trainer_id(self):
        """trainer_id 欠損行でも combined_elo が NaN 連鎖しないこと。

        2026年スクレイプ分は trainer_id 欠損が約48%あり、旧実装では
        combined_elo / combined_elo_rank (重要度1位) が NaN に消えていた。
        欠損コンポーネントは Elo 初期レート1500として合成する。
        """
        df = pd.DataFrame({
            "race_id": ["r1"] * 3,
            "race_date": ["2026-01-01"] * 3,
            "horse_id": ["A", "B", "C"],
            "jockey_id": ["J1", "J2", "J3"],
            "trainer_id": [None, None, None],  # 全欠損
            "horse_number": [1, 2, 3],
            "finish_position": [1, 2, 3],
        })
        fe = FeatureEngineer()
        out = fe._elo_features(df.copy())
        assert out["combined_elo"].notna().all(), "trainer_id欠損でcombined_eloがNaN"
        assert out["combined_elo_rank"].notna().all()


# ----------------------------------------------------------------------
# F5: L2 メタ入力スケールのフラグゲート
# ----------------------------------------------------------------------
class _ConstModel:
    """predict が固定ベクトルを返すスタブ。"""
    def __init__(self, values):
        self._v = np.asarray(values, dtype=float)

    def predict(self, X):
        return self._v[: len(X)]


class _HalfCalibrator:
    def transform(self, p):
        return np.asarray(p, dtype=float) * 0.5


class _EchoL2:
    """meta_X の win列 (index 2) をそのまま返すメタモデル。入力スケール検証用。"""
    def predict(self, meta_X):
        return np.asarray(meta_X)[:, 2]


class TestL2MetaInputScale:
    def _make_model(self, calibrated_flag):
        m = RacePredictionModel()
        m.feature_columns = ["f0", "f1"]
        m.models = {
            "position": _ConstModel([5.0, 6.0, 7.0]),
            "top3": _ConstModel([0.6, 0.5, 0.4]),
            "win": _ConstModel([0.8, 0.6, 0.4]),
        }
        m._calibrators = {"top3": _HalfCalibrator(), "win": _HalfCalibrator()}
        m._l2_position = _EchoL2()
        m._l2_inputs_calibrated = calibrated_flag
        return m

    def _df(self):
        return pd.DataFrame({
            "race_id": ["r", "r", "r"],
            "horse_number": [1, 2, 3],
            "f0": [0.0, 0.0, 0.0],
            "f1": [1.0, 1.0, 1.0],
        })

    def test_new_model_feeds_calibrated_probs_to_l2(self):
        m = self._make_model(calibrated_flag=True)
        res = m.predict(self._df())
        res = res.sort_values("horse_number")
        # L2(=win列エコー) にキャリブレ済み win (0.8*0.5=0.4 ...) が渡ること
        np.testing.assert_allclose(
            res["pred_position"].values, [0.4, 0.3, 0.2], rtol=1e-6
        )

    def test_legacy_model_keeps_raw_probs(self):
        m = self._make_model(calibrated_flag=False)
        res = m.predict(self._df())
        res = res.sort_values("horse_number")
        # 旧挙動: 生の win (0.8, 0.6, 0.4) がそのまま渡る
        np.testing.assert_allclose(
            res["pred_position"].values, [0.8, 0.6, 0.4], rtol=1e-6
        )

    def test_prob_columns_stay_calibrated_regardless(self):
        for flag in (True, False):
            m = self._make_model(calibrated_flag=flag)
            res = m.predict(self._df()).sort_values("horse_number")
            np.testing.assert_allclose(res["prob_win"].values, [0.4, 0.3, 0.2], rtol=1e-6)


# ----------------------------------------------------------------------
# F9: 予測時の欠損補完は学習時中央値を使う (train/serve 整合)
# ----------------------------------------------------------------------
class _EchoFeatureModel:
    """predict が最初の特徴量列をそのまま返すスタブ (補完値の観測用)。"""
    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


class TestFeatureMedianImputation:
    def _df_with_nan(self):
        return pd.DataFrame({
            "race_id": ["r", "r", "r"],
            "horse_number": [1, 2, 3],
            "f0": [np.nan, 2.0, 4.0],
            "f1": [1.0, 1.0, 1.0],
        })

    def _make_model(self, medians):
        m = RacePredictionModel()
        m.feature_columns = ["f0", "f1"]
        m.models = {
            "position": _EchoFeatureModel(),
            "top3": _ConstModel([0.5, 0.5, 0.5]),
            "win": _ConstModel([0.3, 0.3, 0.3]),
        }
        m._feature_medians = medians
        return m

    def test_new_model_uses_train_medians(self):
        """学習時中央値 (f0=10) が保存されていればそれで補完される。"""
        m = self._make_model({"f0": 10.0, "f1": 1.0})
        res = m.predict(self._df_with_nan()).sort_values("horse_number")
        assert res["pred_position"].iloc[0] == pytest.approx(10.0)

    def test_legacy_model_falls_back_to_race_median(self):
        """中央値未保存の旧pklは従来のレース内中央値 (median(2,4)=3) 補完。"""
        m = self._make_model(None)
        res = m.predict(self._df_with_nan()).sort_values("horse_number")
        assert res["pred_position"].iloc[0] == pytest.approx(3.0)


# ----------------------------------------------------------------------
# Ordinal: レース内正規化ラベルの意味論
# ----------------------------------------------------------------------
class TestOrdinalObjective:
    def test_gradient_zero_at_target(self):
        """予測確率がターゲット正規化着順に一致すれば勾配ゼロ。"""
        from model_trainer import _ordinal_regression_lgbm
        obj, _ = _ordinal_regression_lgbm()
        y_norm = np.array([0.25])  # 例: 5頭立ての2着 (2-1)/(5-1)
        raw = np.array([np.log(0.25 / 0.75)])  # sigmoid^-1(0.25)
        grad, hess = obj(raw, _FakeDtrain(y_norm))
        assert abs(grad[0]) < 1e-6
        assert hess[0] > 0

    def test_gradient_direction(self):
        from model_trainer import _ordinal_regression_lgbm
        obj, _ = _ordinal_regression_lgbm()
        # ターゲット0.0 (1着) を p=0.5 と予測 → p を下げる方向 (grad>0)
        grad, _ = obj(np.array([0.0]), _FakeDtrain(np.array([0.0])))
        assert grad[0] > 0
        # ターゲット1.0 (最下位) を p=0.5 と予測 → p を上げる方向 (grad<0)
        grad, _ = obj(np.array([0.0]), _FakeDtrain(np.array([1.0])))
        assert grad[0] < 0


# ----------------------------------------------------------------------
# F6: is_outlier がレース結果 (走破タイム) を使わないこと
# ----------------------------------------------------------------------
class TestOutlierFlags:
    def test_finish_time_does_not_influence_flags(self):
        rng = np.random.default_rng(11)
        n = 300
        base = pd.DataFrame({
            "weight": rng.normal(55, 1.0, n),
            "horse_weight": rng.normal(480, 10, n),
            "weight_diff": rng.normal(0, 3, n),
        })
        with_ft = base.copy()
        with_ft["finish_time_sec"] = rng.normal(95, 1, n)
        with_ft.loc[:10, "finish_time_sec"] = 500.0  # 極端な走破タイム

        proc = RaceDataProcessor()
        flags_a = proc._outlier_flags(base.copy())["is_outlier"].values
        flags_b = proc._outlier_flags(with_ft.copy())["is_outlier"].values
        # 走破タイムの有無・異常値に関わらずフラグが同一 = 結果情報を未使用
        np.testing.assert_array_equal(flags_a, flags_b)


# ----------------------------------------------------------------------
# F7: オッズ派生特徴量のデフォルト除外
# ----------------------------------------------------------------------
class TestOddsFeaturePolicy:
    ODDS_DERIVED = [
        "odds", "popularity", "log_odds", "odds_rank_in_race", "odds_vs_mean",
        "is_favorite", "implied_prob", "normalized_implied_prob", "value_gap",
        "odds_rank", "odds_ratio", "race_avg_odds", "race_min_odds",
    ]

    def _df(self):
        data = {c: [1.0, 2.0] for c in self.ODDS_DERIVED}
        data["horse_avg_pos_3"] = [3.0, 4.0]
        data["distance"] = [1600, 1600]
        return pd.DataFrame(data)

    def test_fe_excludes_all_odds_derivatives_by_default(self):
        cols = FeatureEngineer().get_feature_columns(self._df())
        leaked = [c for c in self.ODDS_DERIVED if c in cols]
        assert not leaked, f"オッズ派生が既定で混入: {leaked}"
        assert "horse_avg_pos_3" in cols and "distance" in cols

    def test_fe_include_odds_opt_in(self):
        cols = FeatureEngineer().get_feature_columns(self._df(), include_odds=True)
        assert "log_odds" in cols and "implied_prob" in cols

    def test_afe_excludes_all_odds_derivatives_by_default(self):
        cols = AdvancedFeatureEngineer().get_extended_feature_columns(self._df(), [])
        leaked = [c for c in self.ODDS_DERIVED if c in cols]
        assert not leaked, f"AFE側でオッズ派生が混入: {leaked}"
        assert "horse_avg_pos_3" in cols
