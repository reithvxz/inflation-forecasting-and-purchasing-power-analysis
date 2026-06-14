import json
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse


class UsdIdrApiTests(TestCase):
    @patch("urllib.request.urlopen")
    def test_uses_latest_two_daily_rates_for_change(self, mock_urlopen):
        payload = {
            "amount": 1.0,
            "base": "USD",
            "start_date": "2026-06-09",
            "end_date": "2026-06-11",
            "rates": {
                "2026-06-09": {"IDR": 17950},
                "2026-06-10": {"IDR": 17910},
                "2026-06-11": {"IDR": 17974},
            },
        }
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode()
        response.__enter__.return_value = response
        mock_urlopen.return_value = response

        api_response = self.client.get(reverse("api_usd_idr_latest"))
        data = api_response.json()

        self.assertEqual(api_response.status_code, 200)
        self.assertEqual(data["latest"], 17974)
        self.assertEqual(data["daily_date"], "2026-06-11")
        self.assertEqual(data["previous_rate"], 17910)
        self.assertEqual(data["previous_date"], "2026-06-10")
        self.assertEqual(data["change_pct"], 0.36)
        self.assertEqual(data["history"], [17950, 17910, 17974])
        self.assertEqual(data["data_type"], "daily")


class HomePageUsdIdrTests(TestCase):
    def test_home_uses_shared_live_exchange_rate_endpoint(self):
        response = self.client.get(reverse("home"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.context["ridge_test_r2"], 0)
        self.assertContains(response, 'id="home-usd-value"')
        self.assertContains(response, 'id="home-usd-change"')
        self.assertContains(response, 'id="home-usd-date"')
        self.assertIn("fetch('/api/usd-idr/')", html)
        self.assertContains(response, "Buka Panduan")


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

    def test_daya_beli_page_renders_basic_and_advanced_modes(self):
        response = self.client.get(reverse("daya_beli"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Basic")
        self.assertContains(response, "Advanced")
        self.assertContains(response, "Provinsi")
        self.assertNotIn("baseValue", html)
        self.assertNotIn("slopePerPercent", html)


class GuideAndDashboardTests(TestCase):
    def test_guide_page_is_accessible_and_appears_in_nav(self):
        response = self.client.get(reverse("guide"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Panduan")
        self.assertIn(reverse("guide"), html)
        self.assertContains(response, "MoM")
        self.assertContains(response, "YoY")
        self.assertContains(response, "Y-to-D")

    def test_dashboard_shows_orientation_panel_and_human_labels(self):
        response = self.client.get(reverse("landing"))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="orientation-panel"')
        self.assertContains(response, "Perubahan harga bulan ini")
        self.assertContains(response, "Perbandingan dengan bulan yang sama tahun lalu")
        self.assertContains(response, "Akumulasi sejak Januari")
        self.assertIn(reverse("guide"), html)


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
        self.assertNotIn("PROVINCE_COORDS", html)
