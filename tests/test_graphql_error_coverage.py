"""Tests to improve GraphQL error path coverage."""

import pytest
from unittest.mock import patch, MagicMock
from openapi_doc_generator.graphql import GraphQLSchema


def test_graphql_introspection_with_errors(tmp_path):
    """Test GraphQL introspection when execution returns errors."""
    schema_file = tmp_path / "schema.graphql"
    schema_file.write_text("""
    type Query {
        hello: String
    }
    """)

    gql_schema = GraphQLSchema(str(schema_file))

    # Mock graphql_sync to return a result with errors
    mock_result = MagicMock()
    mock_result.errors = [Exception("Introspection failed")]
    mock_result.data = None

    with patch("openapi_doc_generator.graphql.graphql_sync", return_value=mock_result):
        with pytest.raises(ValueError, match="Introspection failed"):
            gql_schema.introspect()


def test_graphql_introspection_with_no_data(tmp_path):
    """Test GraphQL introspection when execution returns no data."""
    schema_file = tmp_path / "schema.graphql"
    schema_file.write_text("""
    type Query {
        hello: String
    }
    """)

    gql_schema = GraphQLSchema(str(schema_file))

    # Mock graphql_sync to return a result with no data and no errors
    mock_result = MagicMock()
    mock_result.errors = None
    mock_result.data = None

    with patch("openapi_doc_generator.graphql.graphql_sync", return_value=mock_result):
        with pytest.raises(ValueError, match="Failed to introspect schema"):
            gql_schema.introspect()
