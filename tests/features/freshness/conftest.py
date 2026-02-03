from unittest.mock import MagicMock, patch

import pytest

from src.features.freshness.dtos.freshness_res_dto import FreshnessResDto
from src.features.freshness.freshness_service import (
    FreshnessService,
    get_freshness_service,
)
from src.main import app


@pytest.fixture
def mock_freshness_service():
    mock_service = MagicMock(spec=FreshnessService)
    expected_response = FreshnessResDto(
        label="fresh_apple", confidence=0.99, class_id=0
    )
    mock_service.predict.return_value = expected_response
    return mock_service


@pytest.fixture(autouse=True)
def inject_freshness_mock(mock_freshness_service):
    with patch("src.features.freshness.freshness_service.FreshnessService") as mock_cls:
        mock_cls.return_value = mock_freshness_service
        app.dependency_overrides[get_freshness_service] = lambda: mock_freshness_service
        yield
    app.dependency_overrides.clear()
