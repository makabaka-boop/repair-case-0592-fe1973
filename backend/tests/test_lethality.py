"""致死量计算边界测试：速率、梯形积分、未舍入累加、四舍五入与放行门槛。"""

from decimal import Decimal

import pytest

from app.lethality import compute_lethality, lethality_rate, round_half_up


class TestLethalityRate:
    def test_reference_temperature_gives_unit_rate(self):
        assert lethality_rate(121.1) == pytest.approx(1.0)

    def test_ten_degrees_up_multiplies_rate_by_ten(self):
        assert lethality_rate(131.1) == pytest.approx(10.0)

    def test_ten_degrees_down_divides_rate_by_ten(self):
        assert lethality_rate(111.1) == pytest.approx(0.1)

    def test_boundaries_of_allowed_range(self):
        assert lethality_rate(100.0) == pytest.approx(10.0 ** -2.11)
        assert lethality_rate(140.0) == pytest.approx(10.0 ** 1.89)


class TestRoundHalfUp:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (2.994, "2.99"),
            (2.995, "3.00"),  # 临界值：四舍五入向上进位
            (3.004, "3.00"),
            (3.005, "3.01"),
            (2.985, "2.99"),
            (0.0077625, "0.01"),
        ],
    )
    def test_half_up_at_boundaries(self, value, expected):
        assert round_half_up(value) == Decimal(expected)


class TestComputeLethality:
    def test_constant_reference_temperature_180s_exactly_passes(self):
        result = compute_lethality(
            [(0, 121.1), (60, 121.1), (120, 121.1), (180, 121.1)]
        )
        assert result.f0 == Decimal("3.00")
        assert result.passed is True
        assert result.shortfall is None
        assert len(result.segments) == 3
        for segment in result.segments:
            assert segment.contribution == pytest.approx(1.0)

    def test_179s_fails_and_reports_shortfall(self):
        result = compute_lethality(
            [(0, 121.1), (60, 121.1), (120, 121.1), (179, 121.1)]
        )
        assert result.f0 == Decimal("2.98")
        assert result.passed is False
        assert result.shortfall == Decimal("0.02")

    def test_trapezoid_averages_endpoint_rates(self):
        # 100→140 °C 持续 60 s：贡献 = (r100 + r140) / 2 × 1 min
        result = compute_lethality([(0, 100.0), (60, 140.0)])
        r100 = 10.0 ** ((100.0 - 121.1) / 10.0)
        r140 = 10.0 ** ((140.0 - 121.1) / 10.0)
        assert result.total_unrounded == pytest.approx((r100 + r140) / 2.0)
        only = result.segments[0]
        assert only.duration_seconds == 60
        assert only.start_rate == pytest.approx(r100)
        assert only.end_rate == pytest.approx(r140)

    def test_segments_accumulate_unrounded_before_final_rounding(self):
        # 每段约 0.00388 min：若逐段先保留两位小数再相加会得到 0.00，
        # 规范要求未舍入累加后整体四舍五入，应为 0.01。
        result = compute_lethality([(0, 100.0), (30, 100.0), (60, 100.0)])
        for segment in result.segments:
            assert segment.contribution == pytest.approx(0.00388, abs=1e-5)
        assert result.f0 == Decimal("0.01")

    def test_segment_contributions_sum_to_unrounded_total(self):
        points = [(0, 110.0), (20, 115.5), (45, 121.1), (60, 130.0)]
        result = compute_lethality(points)
        assert sum(s.contribution for s in result.segments) == pytest.approx(
            result.total_unrounded
        )

    def test_fewer_than_two_points_rejected(self):
        with pytest.raises(ValueError):
            compute_lethality([(0, 121.1)])
