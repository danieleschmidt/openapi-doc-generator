import json
import pytest
from openapi_doc_generator.graphql import GraphQLSchema
from openapi_doc_generator.cli import main


def test_graphql_introspection(tmp_path, capsys):
    schema = tmp_path / "schema.graphql"
    schema.write_text("""type Query { hello: String }""")
    result = GraphQLSchema(str(schema)).introspect()
    assert "__schema" in result

    main(["--app", str(schema), "--format", "graphql"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "__schema" in data


class TestGraphQLErrorHandling:
    """Test GraphQL error scenarios and edge cases."""
    
    def test_graphql_schema_file_not_found(self):
        """Test behavior when GraphQL schema file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            GraphQLSchema("/nonexistent/schema.graphql")
    
    def test_graphql_invalid_schema_syntax(self, tmp_path):
        """Test behavior with syntactically invalid GraphQL schema."""
        schema = tmp_path / "invalid.graphql"
        schema.write_text("""
        type Query {
            invalid syntax here
            missing closing brace
        """)
        
        gql_schema = GraphQLSchema(str(schema))
        # Should raise ValueError due to schema syntax errors
        with pytest.raises(ValueError):
            gql_schema.introspect()
    
    def test_graphql_empty_schema_file(self, tmp_path):
        """Test behavior with empty GraphQL schema file."""
        schema = tmp_path / "empty.graphql"
        schema.write_text("")
        
        gql_schema = GraphQLSchema(str(schema))
        # Should raise ValueError due to invalid schema
        with pytest.raises(ValueError):
            gql_schema.introspect()
    
    def test_graphql_schema_with_execution_errors(self, tmp_path):
        """Test handling of GraphQL execution errors during introspection."""
        schema = tmp_path / "problematic.graphql"
        # Create a schema that might cause execution issues
        schema.write_text("""
        type Query {
            field: ProblematicType
        }
        
        # Missing type definition for ProblematicType will cause issues
        """)
        
        gql_schema = GraphQLSchema(str(schema))
        # Should raise ValueError due to execution errors
        with pytest.raises(ValueError):
            gql_schema.introspect()
    
    def test_graphql_schema_with_io_error(self, tmp_path):
        """Test behavior when schema file causes I/O errors during reading.""" 
        schema = tmp_path / "problematic.graphql"
        schema.write_text("type Query { hello: String }")
        
        # Test by trying to read from a directory instead of a file
        schema.unlink()  # Remove the file
        schema.mkdir()   # Create a directory with the same name
        
        gql_schema = GraphQLSchema(str(schema))
        # Should raise ValueError when trying to read directory as file
        with pytest.raises(ValueError) as exc_info:
            gql_schema.introspect()
        
        assert "Failed to read schema file" in str(exc_info.value)
    
    def test_graphql_cli_with_invalid_schema(self, tmp_path):
        """Test CLI behavior with invalid GraphQL schema."""
        schema = tmp_path / "bad.graphql"
        schema.write_text("invalid graphql syntax {}")
        
        with pytest.raises(SystemExit):
            main(["--app", str(schema), "--format", "graphql"])


class TestGraphQLEdgeCases:
    """Test edge cases in GraphQL handling."""
    
    def test_graphql_very_large_schema(self, tmp_path):
        """Test performance with large GraphQL schema."""
        schema = tmp_path / "large.graphql"
        
        # Generate a large but valid schema
        large_schema = "type Query {\n"
        for i in range(100):
            large_schema += f"  field{i}: String\n"
        large_schema += "}\n"
        
        schema.write_text(large_schema)
        gql_schema = GraphQLSchema(str(schema))
        result = gql_schema.introspect()
        
        # Should handle large schemas without issues
        assert "__schema" in result
        assert "types" in result["__schema"]
    
    def test_graphql_schema_with_comments(self, tmp_path):
        """Test schema with GraphQL comments."""
        schema = tmp_path / "commented.graphql"
        schema.write_text("""
        # This is a comment
        type Query {
            # Another comment
            hello: String # Field comment
        }
        """)
        
        gql_schema = GraphQLSchema(str(schema))
        result = gql_schema.introspect()
        assert "__schema" in result
    
    def test_graphql_schema_with_unicode(self, tmp_path):
        """Test schema with Unicode characters."""
        schema = tmp_path / "unicode.graphql"
        schema.write_text("""
        type Query {
            # 测试中文注释
            greeting: String
            # éñçødÿñg test
            encoded: String
        }
        """, encoding='utf-8')
        
        gql_schema = GraphQLSchema(str(schema))
        result = gql_schema.introspect()
        assert "__schema" in result
