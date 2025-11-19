"""
Alert manager for market monitoring.

Creates and manages alerts for interesting market finds.
"""
from typing import List, Literal

from backend.core.config import settings
from backend.core.database import db
from backend.core.logging import log
from backend.core.models import Alert, AnalysisResult, MarketItem


class AlertManager:
    """
    Manages alerts for market monitoring.

    Evaluates items against alert rules and sends notifications.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.enabled = settings.ALERT_ENABLED
        self.bargain_threshold = settings.ALERT_BARGAIN_THRESHOLD
        self.min_authenticity = settings.ALERT_MIN_AUTHENTICITY_SCORE
        self.rare_styles = settings.alert_rare_styles_list

    def evaluate_item(
        self,
        market_item: MarketItem,
        analysis_result: AnalysisResult
    ) -> List[Alert]:
        """
        Evaluate a market item and create alerts if criteria are met.

        Args:
            market_item: Market item
            analysis_result: Analysis result for the item

        Returns:
            List of created alerts
        """
        if not self.enabled or not market_item.id:
            return []

        alerts = []

        # Check for bargain (undervalued)
        bargain_alert = self._check_bargain(market_item, analysis_result)
        if bargain_alert:
            alerts.append(bargain_alert)

        # Check for rare style
        rare_style_alert = self._check_rare_style(market_item, analysis_result)
        if rare_style_alert:
            alerts.append(rare_style_alert)

        # Check for high authenticity + good price
        quality_alert = self._check_quality_find(market_item, analysis_result)
        if quality_alert:
            alerts.append(quality_alert)

        # Save alerts to database
        for alert in alerts:
            alert_id = db.add_alert(alert)
            log.info(f"Created alert {alert_id}: {alert.reason}")

        return alerts

    def _check_bargain(
        self,
        market_item: MarketItem,
        analysis_result: AnalysisResult
    ) -> Optional[Alert]:
        """
        Check if item is significantly underpriced.

        Args:
            market_item: Market item
            analysis_result: Analysis result

        Returns:
            Alert if bargain detected, None otherwise
        """
        estimated_value = analysis_result.valuation.estimated_value
        listed_price = market_item.price

        if estimated_value <= 0 or listed_price <= 0:
            return None

        # Check if price is less than threshold of estimated value
        price_ratio = listed_price / estimated_value

        if price_ratio < self.bargain_threshold:
            return Alert(
                item_id=market_item.id,
                alert_level="critical",
                reason=f"Potential bargain: Listed at €{listed_price:,.0f}, estimated value €{estimated_value:,.0f} ({price_ratio:.1%} of estimated value)"
            )

        return None

    def _check_rare_style(
        self,
        market_item: MarketItem,
        analysis_result: AnalysisResult
    ) -> Optional[Alert]:
        """
        Check if item is a rare or collectible style.

        Args:
            market_item: Market item
            analysis_result: Analysis result

        Returns:
            Alert if rare style, None otherwise
        """
        epoch = analysis_result.style_estimation.epoch

        if epoch in self.rare_styles:
            return Alert(
                item_id=market_item.id,
                alert_level="warning",
                reason=f"Rare epoch detected: {epoch}"
            )

        return None

    def _check_quality_find(
        self,
        market_item: MarketItem,
        analysis_result: AnalysisResult
    ) -> Optional[Alert]:
        """
        Check for high-quality find (high authenticity + reasonable price).

        Args:
            market_item: Market item
            analysis_result: Analysis result

        Returns:
            Alert if quality find, None otherwise
        """
        authenticity = analysis_result.authenticity_score
        estimated_value = analysis_result.valuation.estimated_value
        listed_price = market_item.price

        # High authenticity and reasonable price
        if authenticity >= 80 and estimated_value > 5000 and listed_price < estimated_value:
            return Alert(
                item_id=market_item.id,
                alert_level="warning",
                reason=f"High-quality find: Authenticity {authenticity}/100, estimated value €{estimated_value:,.0f}"
            )

        return None

    def send_notification(self, alert: Alert, market_item: MarketItem):
        """
        Send notification for an alert.

        Args:
            alert: Alert to notify about
            market_item: Associated market item
        """
        if not self.enabled:
            return

        # Email notification
        if settings.ALERT_EMAIL_ENABLED:
            self._send_email(alert, market_item)

        # Telegram notification
        if settings.ALERT_TELEGRAM_ENABLED:
            self._send_telegram(alert, market_item)

        # Mark alert as notified
        db.mark_alert_notified(alert.id)

    def _send_email(self, alert: Alert, market_item: MarketItem):
        """
        Send email notification.

        Args:
            alert: Alert
            market_item: Market item
        """
        # Placeholder - would implement SMTP email sending
        log.info(f"Email notification (not implemented): {alert.reason}")

    def _send_telegram(self, alert: Alert, market_item: MarketItem):
        """
        Send Telegram notification.

        Args:
            alert: Alert
            market_item: Market item
        """
        # Placeholder - would implement Telegram bot API
        log.info(f"Telegram notification (not implemented): {alert.reason}")

    def get_pending_alerts(self, limit: int = 100) -> List[dict]:
        """
        Get alerts that haven't been notified yet.

        Args:
            limit: Maximum alerts to retrieve

        Returns:
            List of alert dictionaries
        """
        return db.get_alerts(notified=False, limit=limit)


# Global alert manager instance
alert_manager = AlertManager()
