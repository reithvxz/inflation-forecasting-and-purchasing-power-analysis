import json
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse


class UsdIdrApiTests(TestCase):
    @patch("urllib.request.urlopen")
    def test_uses_month_start_rate_for_change(self, mock_urlopen):
        latest_payload = {
            "result": "success",
            "time_last_update_unix": 1781481600,
            "rates": {
                "IDR": 17831.64,
            },
        }
        history_payload = {
            "amount": 1.0,
            "base": "USD",
            "start_date": "2026-06-09",
            "end_date": "2026-06-12",
            "rates": {
                "2026-06-10": {"IDR": 17950},
                "2026-06-11": {"IDR": 17910},
                "2026-06-12": {"IDR": 17788},
            },
        }

        latest_response = MagicMock()
        latest_response.read.return_value = json.dumps(latest_payload).encode()
        latest_response.__enter__.return_value = latest_response

        history_response = MagicMock()
        history_response.read.return_value = json.dumps(history_payload).encode()
        history_response.__enter__.return_value = history_response

        mock_urlopen.side_effect = [latest_response, history_response]

        api_response = self.client.get(reverse("api_usd_idr_latest"))
        data = api_response.json()

        self.assertEqual(api_response.status_code, 200)
        self.assertEqual(data["latest"], 17831.64)
        self.assertEqual(data["daily_date"], "2026-06-15")
        self.assertEqual(data["previous_rate"], 17950.0)
        self.assertEqual(data["previous_date"], "2026-06-10")
        self.assertEqual(data["month_start_rate"], 17950.0)
        self.assertEqual(data["month_start_date"], "2026-06-10")
        self.assertEqual(data["change_pct"], -0.66)
        self.assertEqual(data["history"], [17950.0, 17910.0, 17788.0])
        self.assertEqual(data["data_type"], "daily")
        self.assertIn("no-store", api_response["Cache-Control"])


class HomePageUsdIdrTests(TestCase):
    def test_home_uses_shared_live_exchange_rate_endpoint(self):
        response = self.client.get(reverse("home"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.context["ridge_test_r2"], 0)
        self.assertIn("inflasi_pred", response.context)
        self.assertIn("inflasi_model_name", response.context)
        self.assertContains(response, 'id="home-usd-value"')
        self.assertContains(response, 'id="home-usd-change"')
        self.assertContains(response, 'id="home-usd-date"')
        self.assertContains(response, "R^2 uji model daya beli riil")
        self.assertNotContains(response, "Akurasi Model")
        self.assertNotIn(">0.55%</div>", html)
        self.assertIn("fetch('/api/usd-idr/', {", html)
        self.assertIn("cache: 'no-store'", html)
        self.assertContains(response, "Buka Daya Beli")

    def test_home_and_landing_use_same_primary_inflation_forecast(self):
        home_response = self.client.get(reverse("home"))
        landing_response = self.client.get(reverse("landing"))

        self.assertEqual(home_response.status_code, 200)
        self.assertEqual(landing_response.status_code, 200)
        self.assertAlmostEqual(
            float(home_response.context["inflasi_pred"]),
            float(landing_response.context["inflasi_pred"]),
            places=6,
        )
        self.assertEqual(
            home_response.context["inflasi_model_name"],
            landing_response.context["inflasi_model_name"],
        )

    def test_forecasting_page_embeds_multi_horizon_payload(self):
        response = self.client.get(reverse("forecasting"))
        payload = json.loads(response.context["forecast_payload_json"])

        self.assertEqual(response.status_code, 200)
        self.assertIn("horizons", payload)
        self.assertIn("1m", payload["horizons"])
        self.assertAlmostEqual(
            float(payload["horizons"]["1m"]["headline_forecast"]),
            float(payload["horizons"]["1m"]["top_models"][0]["point_forecast"]),
            places=6,
        )

    def test_inflation_forecast_api_returns_multi_horizon_contract(self):
        response = self.client.get(reverse("api_inflation_forecast"))
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("generated_at", data)
        self.assertIn("history", data)
        self.assertIn("horizons", data)
        self.assertEqual(sorted(data["horizons"].keys()), ["12m", "1m", "3m", "6m"])
        one_month = data["horizons"]["1m"]
        self.assertEqual(len(one_month["top_models"]), 2)
        self.assertIn("headline_model", one_month)
        self.assertIn("headline_forecast", one_month)
        self.assertIn("risk_note", one_month)


class DayaBeliSimulationTests(TestCase):
    def test_simulate_daya_beli_requires_province_and_returns_positive_value(self):
        response = self.client.get(reverse("api_simulate"), {"provinsi": "Aceh", "inflasi": 0})
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["province"], "Aceh")
        self.assertGreater(data["predicted_pengeluaran"], 800000)
        self.assertIn("baseline_year", data)
        self.assertIn("inputs_used", data)

    def test_simulate_daya_beli_changes_when_province_changes(self):
        aceh = self.client.get(reverse("api_simulate"), {"provinsi": "Aceh", "inflasi": 2.5}).json()
        bali = self.client.get(reverse("api_simulate"), {"provinsi": "Bali", "inflasi": 2.5}).json()

        self.assertNotEqual(aceh["predicted_pengeluaran"], bali["predicted_pengeluaran"])

    def test_simulate_daya_beli_changes_when_core_inputs_change(self):
        baseline = self.client.get(reverse("api_simulate"), {"provinsi": "Aceh", "inflasi": 2.5}).json()
        adjusted = self.client.get(
            reverse("api_simulate"),
            {
                "provinsi": "Aceh",
                "inflasi": 2.5,
                "ump": 4200000,
                "tpt": 4.2,
                "pdrb_hargakonstan": 52000,
            },
        ).json()

        self.assertNotEqual(baseline["predicted_pengeluaran"], adjusted["predicted_pengeluaran"])

    def test_simulate_daya_beli_supports_indonesia_baseline(self):
        response = self.client.get(reverse("api_simulate"), {"provinsi": "Indonesia", "inflasi": 2.5})
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["province"], "Indonesia")
        self.assertGreater(data["predicted_pengeluaran"], 800000)
        self.assertEqual(data["inputs_used"]["inflasi"], 2.5)

    def test_simulate_daya_beli_high_inflation_scenario_stays_in_realistic_band(self):
        baseline = self.client.get(reverse("api_simulate"), {"provinsi": "Indonesia", "inflasi": 2.5}).json()
        stressed = self.client.get(reverse("api_simulate"), {"provinsi": "Indonesia", "inflasi": 6.0}).json()

        self.assertGreater(stressed["predicted_pengeluaran"], 1000000)
        self.assertLess(stressed["predicted_pengeluaran"], baseline["predicted_pengeluaran"])
        self.assertEqual(stressed["inputs_used"]["inflasi"], 6.0)

    def test_simulate_requires_wilayah_selection_message(self):
        response = self.client.get(reverse("api_simulate"))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Wilayah wajib dipilih")

    def test_daya_beli_page_renders_basic_and_advanced_modes(self):
        response = self.client.get(reverse("daya_beli"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Basic")
        self.assertContains(response, "Advanced")
        self.assertContains(response, "Wilayah")
        self.assertContains(response, "Indonesia")
        self.assertContains(response, "Simulasi pengeluaran riil per kapita untuk membaca proksi daya beli")
        self.assertContains(response, "pengeluaran riil per kapita per bulan")
        self.assertContains(response, "Interpretasi utama")
        self.assertContains(response, "Batasan model")
        self.assertContains(response, "proksi daya beli")
        self.assertNotIn("baseValue", html)
        self.assertNotIn("slopePerPercent", html)


class GuideAndDashboardTests(TestCase):
    def test_guide_page_is_accessible_and_appears_in_nav(self):
        response = self.client.get(reverse("guide"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Panduan")
        self.assertIn(reverse("guide"), html)
        self.assertContains(response, "Panduan Pembacaan")
        self.assertContains(response, "Panduan Teknis")
        self.assertContains(response, "MoM")
        self.assertContains(response, "YoY")
        self.assertContains(response, "Y-to-D")
        self.assertContains(response, "Keluarga model inflasi yang dipakai")
        self.assertContains(response, "ARIMA")
        self.assertContains(response, "LSTM / Bi-LSTM")
        self.assertContains(response, "Fitur inti SARIMAX untuk model inflasi")
        self.assertContains(response, "Audit kontribusi fitur SARIMAX")
        self.assertContains(response, "Ringkasan teknis model proksi daya beli")

    def test_global_nav_prioritizes_forecasting_and_daya_beli_before_dashboard(self):
        response = self.client.get(reverse("guide"))
        html = response.content.decode()

        self.assertLess(html.index("<span>Forecasting</span>"), html.index("<span>Daya Beli</span>"))
        self.assertLess(html.index("<span>Daya Beli</span>"), html.index("<span>Dashboard</span>"))

    def test_dashboard_shows_orientation_panel_and_human_labels(self):
        response = self.client.get(reverse("landing"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="orientation-panel"')
        self.assertContains(response, "Perubahan harga bulan ini")
        self.assertContains(response, "Perbandingan dengan bulan yang sama tahun lalu")
        self.assertContains(response, "Akumulasi sejak Januari")
        self.assertContains(response, "R^2 uji model daya beli riil")
        self.assertNotContains(response, "Akurasi model daya beli")
        self.assertContains(response, "Estimasi pengeluaran riil per kapita")
        self.assertNotContains(response, "Kalau ingin baca cepat")
        self.assertNotContains(response, "bahasa yang lebih santai")
        self.assertEqual(response.context["province_count"], 38)
        self.assertContains(response, "Model utama")
        self.assertIn(reverse("guide"), html)

    def test_home_uses_actual_province_count_without_indonesia_aggregate(self):
        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["province_count"], 38)
        self.assertContains(response, "Baseline wilayah terbaru dipakai untuk simulasi dan peta ekonomi.")

    def test_inflation_summary_api_uses_no_store_headers(self):
        response = self.client.get(reverse("api_inflasi_summary"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response["Cache-Control"])

    def test_inflation_summary_api_includes_date_aliases(self):
        response = self.client.get(reverse("api_inflasi_summary"))
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("as_of", data)
        self.assertEqual(data["date"], data["as_of"])


class ComparisonAndScenarioTests(TestCase):
    def test_compare_page_renders_workspace_controls(self):
        response = self.client.get(reverse("compare"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="prov1"', html)
        self.assertIn('id="metric"', html)
        self.assertIn('id="compareChart"', html)
        self.assertIn('id="rankingList"', html)
        self.assertIn('id="radarChart"', html)

    def test_scenarios_page_uses_api_backed_analysis(self):
        response = self.client.get(reverse("scenarios"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/scenario-analysis/", html)
        self.assertNotIn("Math.random", html)
        self.assertContains(response, "Scenario Analysis")
        self.assertContains(response, "proksi daya beli")

    def test_scenario_analysis_api_returns_deterministic_contract(self):
        response = self.client.get(reverse("api_scenario_analysis"), {"scenario_id": "inflation_shock"})
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["scenario_id"], "inflation_shock")
        self.assertIn("baseline_value", data)
        self.assertIn("scenario_value", data)
        self.assertIn("change_pct", data)
        self.assertIn("series", data)
        self.assertIn("province_impacts", data)
        self.assertGreaterEqual(len(data["province_impacts"]), 10)
        self.assertIn("no-store", response["Cache-Control"])

    def test_province_apis_exclude_indonesia_from_provincial_surfaces(self):
        province_list = self.client.get(reverse("api_province_list")).json()
        metrics_latest = self.client.get(reverse("api_metrics_latest")).json()

        self.assertNotIn("Indonesia", province_list["provinces"])
        self.assertNotIn("Indonesia", metrics_latest["provinces"])


class EconomicMapPageTests(TestCase):
    def test_map_uses_province_polygons_and_insight_panel(self):
        response = self.client.get(reverse("map"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertIn("indonesia_provinces.geojson", html)
        self.assertIn("L.geoJSON", html)
        self.assertIn('id="mapSummary"', html)
        self.assertIn('id="provinceRanking"', html)
        self.assertIn('id="resetMapView"', html)
        self.assertIn('id="mapYear"', html)
        self.assertNotIn("PROVINCE_COORDS", html)

    def test_metrics_latest_api_returns_latest_year_without_param(self):
        response = self.client.get(reverse("api_metrics_latest"))
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["latest_year"], 2025)
        self.assertEqual(data["selected_year"], 2025)
        self.assertEqual(data["available_years"], [2021, 2022, 2023, 2024, 2025])
        self.assertGreaterEqual(len(data["provinces"]), 35)

    def test_metrics_latest_api_accepts_year_parameter(self):
        latest_response = self.client.get(reverse("api_metrics_latest"))
        selected_response = self.client.get(reverse("api_metrics_latest"), {"year": 2023})

        latest_payload = latest_response.json()
        selected_payload = selected_response.json()

        self.assertEqual(selected_response.status_code, 200)
        self.assertEqual(selected_payload["selected_year"], 2023)
        self.assertEqual(selected_payload["latest_year"], 2025)
        self.assertEqual(selected_payload["available_years"], [2021, 2022, 2023, 2024, 2025])
        self.assertNotEqual(selected_payload["provinces"], latest_payload["provinces"])

    def test_metrics_latest_api_includes_nominal_metric_and_coverage_metadata(self):
        response = self.client.get(reverse("api_metrics_latest"))
        data = response.json()

        self.assertEqual(response.status_code, 200)
        sample_row = next(iter(data["provinces"].values()))
        self.assertIn("Total_Pengeluaran_Riil", sample_row)
        self.assertIn("Total_Pengeluaran", sample_row)
        self.assertIn("coverage_count", data)
        self.assertIn("coverage_total", data)
        self.assertIn("missing_provinces", data)
        self.assertLessEqual(data["coverage_count"], data["coverage_total"])
