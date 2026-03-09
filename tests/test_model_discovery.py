"""Tests for dynamic local model discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fangbot.gateway.models_catalog import ModelInfo, discover_local_models


class TestDiscoverLocalModels:
    @pytest.mark.asyncio
    async def test_returns_models_from_server(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "llama3.2", "object": "model"},
                {"id": "mistral", "object": "model"},
            ]
        }

        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            models = await discover_local_models("http://localhost:11434/v1")

        assert len(models) == 2
        assert models[0].id == "llama3.2"
        assert models[0].category == "local"
        assert models[1].id == "mistral"

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_connection_error(self):
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=Exception("Connection refused"),
        ):
            models = await discover_local_models("http://localhost:11434/v1")

        assert models == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_non_200(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}

        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            models = await discover_local_models("http://localhost:11434/v1")

        assert models == []
