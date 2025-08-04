"""
Email sender for critical alerts.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
from datetime import datetime

from config.settings import get_settings
from utils.logger import setup_logger


class EmailSender:
    """Send email alerts for critical inventory items."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = setup_logger(self.__class__.__name__)
    
    def send_critical_alerts(self, urgent_orders: List[Dict[str, Any]]) -> bool:
        """
        Send email alerts for urgent orders.
        
        Args:
            urgent_orders: List of urgent order dictionaries
            
        Returns:
            True if email sent successfully
        """
        if not urgent_orders:
            self.logger.info("No urgent orders to send")
            return True
        
        try:
            # Create email content
            subject = f"URGENT: Inventory Alert - {len(urgent_orders)} Items Need Ordering"
            body = self._create_email_body(urgent_orders)
            
            # Send email
            self._send_email(subject, body)
            
            self.logger.info(f"Alert email sent for {len(urgent_orders)} items")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def _create_email_body(self, urgent_orders: List[Dict[str, Any]]) -> str:
        """Create HTML email body."""
        html = f"""
        <html>
        <body>
            <h2>Urgent Inventory Orders Required</h2>
            <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <p>The following items require immediate ordering:</p>
            
            <table border="1" cellpadding="5" cellspacing="0">
                <thead>
                    <tr>
                        <th>Item ID</th>
                        <th>Total Demand</th>
                        <th>Safety Stock</th>
                        <th>Order By</th>
                        <th>Priority</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for order in urgent_orders:
            html += f"""
                    <tr>
                        <td>{order['item_id']}</td>
                        <td>{order['total_demand']:,.0f}</td>
                        <td>{order['safety_stock']:,.0f}</td>
                        <td>{order['order_by_date']}</td>
                        <td style="color: red; font-weight: bold;">{order['priority']}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <p>Please take immediate action to prevent stockouts.</p>
            
            <p>This is an automated message from the Inventory Forecasting System.</p>
        </body>
        </html>
        """
        
        return html
    
    def _send_email(self, subject: str, body: str):
        """Send email via SMTP."""
        if not self.settings.EMAIL_PASSWORD:
            self.logger.warning("Email password not configured, skipping email send")
            return
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.settings.EMAIL_FROM
        msg['To'] = ', '.join(self.settings.EMAIL_TO)
        
        # Attach HTML body
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        with smtplib.SMTP(self.settings.SMTP_HOST, self.settings.SMTP_PORT) as server:
            if self.settings.SMTP_USE_TLS:
                server.starttls()
            server.login(self.settings.EMAIL_FROM, self.settings.EMAIL_PASSWORD)
            server.send_message(msg)