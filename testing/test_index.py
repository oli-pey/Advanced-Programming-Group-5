import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from nicegui import ui, app

from web.index import LandingPage


def test_tc005_empty_canvas_guard():
    """
    TC_005: Verify that clicking 'Predict & Save' without drawing
    displays a warning notification.
    """
    # Create landing page instance
    page = LandingPage()

    # Create mock for ii (interactive image)
    page.ii = Mock()
    page.ii.content = ""  # Empty canvas

    # Create a mock for ui.notify to capture calls
    with patch('web.index.ui.notify') as mock_notify:
        # Synchronously call process_drawing (the async check happens first)
        import asyncio
        asyncio.run(page.process_drawing())

        # Assert notification was called with warning
        mock_notify.assert_called_with(
            "Please draw something first!",
            type='warning'
        )


def test_tc006_session_authentication_guard(monkeypatch):
    """
    TC_006: Verify that clicking 'Predict & Save' without an active session
    displays an authentication error notification.
    """
    # Create landing page instance
    page = LandingPage()

    # Create mock for ii with some content (to pass empty canvas check)
    page.ii = Mock()
    page.ii.content = '<path d="M 10 10 L 20 20"/>'

    # Create a mock storage that returns empty (no user_id)
    mock_user = Mock()
    mock_user.get.return_value = None

    mock_storage = Mock()
    mock_storage.user = mock_user

    # Create a mock for ui.notify to capture calls
    with patch('web.index.ui.notify') as mock_notify:
        # Monkeypatch the storage to avoid NiceGUI initialization
        monkeypatch.setattr('web.index.app.storage', mock_storage)

        # Synchronously call process_drawing
        import asyncio
        asyncio.run(page.process_drawing())

        # Assert notification was called with session error
        mock_notify.assert_called_with(
            "Session expired. Please log in.",
            type='negative'
        )
