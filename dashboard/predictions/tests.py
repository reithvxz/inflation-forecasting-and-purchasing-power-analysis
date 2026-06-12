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

        self.assertContains(response, 'id="home-usd-value"')
        self.assertContains(response, 'id="home-usd-change"')
        self.assertContains(response, 'id="home-usd-date"')
        self.assertIn("fetch('/api/usd-idr/')", html)
        self.assertIn("cdn.jsdelivr.net/npm/chart.js", html)
        self.assertEqual(html.count("drawSpark('spark-inflasi'"), 1)
