# Integration Tests

These tests require a real TWS or IB Gateway connection.

## Setup

1. Start TWS or IB Gateway
2. Use port 7497 for paper trading (recommended)
3. Enable API connections in TWS settings

## Running Tests

```bash
# Run all integration tests
uv run pytest tests/integration/ -v

# Run specific test
uv run pytest tests/integration/test_ibkr_client_integration.py::test_connection_lifecycle -v

# Quick test to verify connection
uv run python test_clean_ibkr_client.py
```

## Test Coverage

- ✅ Connection lifecycle (connect, disconnect, reconnect)
- ✅ Context manager usage
- ✅ Account data retrieval
- ✅ Position retrieval
- ✅ Open orders
- ✅ Multiple reconnection cycles
- ✅ Error handling
- ✅ Concurrent operations
- ⚠️ Order placement (skipped by default - remove skip to test)

## Important Notes

- These tests use paper trading account (port 7497)
- Client ID 99 is used to avoid conflicts
- Order placement test is skipped by default for safety