import textwrap
import pytest
from unittest.mock import patch
from openapi_doc_generator.schema import SchemaInferer


class TestSchemaInferenceErrorHandling:
    """Test error handling scenarios in schema inference."""
    
    def test_schema_inference_file_not_found(self):
        """Test handling when schema file doesn't exist."""
        # This should raise FileNotFoundError before reaching inference error paths
        with pytest.raises(FileNotFoundError):
            SchemaInferer("/nonexistent/file.py").infer()
    
    def test_schema_inference_unreadable_file(self, tmp_path):
        """Test handling of files that can't be read due to I/O errors."""
        file = tmp_path / "models.py"
        file.write_text("class Model: pass")
        
        # Mock pathlib.Path.read_text at module level
        with patch('pathlib.Path.read_text', side_effect=OSError("Permission denied")):
            inferer = SchemaInferer(str(file))
            schemas = inferer.infer()
            
        # Should return empty list instead of crashing
        assert schemas == []
    
    def test_schema_inference_unicode_decode_error(self, tmp_path):
        """Test handling of files with invalid encoding.""" 
        file = tmp_path / "models.py"
        file.write_text("class Model: pass")
        
        # Mock pathlib.Path.read_text at module level
        with patch('pathlib.Path.read_text', side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")):
            inferer = SchemaInferer(str(file))
            schemas = inferer.infer()
            
        # Should return empty list instead of crashing
        assert schemas == []
    
    def test_schema_inference_syntax_error(self, tmp_path):
        """Test handling of Python files with syntax errors."""
        file = tmp_path / "models.py"
        # Write syntactically invalid Python code
        file.write_text("""
        class Model
            invalid syntax here
            missing colons and proper indentation
        """)
        
        inferer = SchemaInferer(str(file))
        schemas = inferer.infer()
        
        # Should return empty list instead of crashing
        assert schemas == []
    
    def test_schema_inference_invalid_python_structure(self, tmp_path):
        """Test handling of files with invalid Python structure."""
        file = tmp_path / "models.py"
        # Write code that parses but has structural issues
        file.write_text("""
        def incomplete_function(
            # Incomplete function definition
        
        class Model:
            field: unclosed_parenthesis(
        """)
        
        inferer = SchemaInferer(str(file))
        schemas = inferer.infer()
        
        # Should handle syntax errors gracefully
        assert schemas == []
    
    def test_schema_inference_class_processing_exception(self, tmp_path):
        """Test handling of exceptions during class processing."""
        file = tmp_path / "models.py"
        file.write_text("""
        from dataclasses import dataclass
        
        @dataclass  
        class ValidModel:
            id: int
            name: str
        """)
        
        inferer = SchemaInferer(str(file))
        
        # Mock _process_class to raise an exception for this specific class
        original_process_class = inferer._process_class
        def mock_process_class(node):
            if node.name == "ValidModel":
                raise RuntimeError("Simulated processing failure")
            return original_process_class(node)
        
        with patch.object(inferer, '_process_class', side_effect=mock_process_class):
            schemas = inferer.infer()
        
        # Should handle exception and return empty list (since our only class failed)
        assert schemas == []
    
    def test_schema_inference_partial_class_processing_failure(self, tmp_path):
        """Test that one failed class doesn't prevent processing others."""
        file = tmp_path / "models.py"
        file.write_text(textwrap.dedent("""
            from dataclasses import dataclass
            
            @dataclass
            class GoodModel:
                id: int
                
            @dataclass  
            class ProblematicModel:
                field: int
                
            @dataclass
            class AnotherGoodModel:
                name: str
        """).strip())
        
        inferer = SchemaInferer(str(file))
        
        # Mock _process_class to fail only for ProblematicModel
        original_process_class = inferer._process_class
        def mock_process_class(node):
            if node.name == "ProblematicModel":
                raise ValueError("Simulated processing failure")
            return original_process_class(node)
        
        with patch.object(inferer, '_process_class', side_effect=mock_process_class):
            schemas = inferer.infer()
        
        # Should successfully process the other two models
        assert len(schemas) == 2
        schema_names = {schema.name for schema in schemas}
        assert schema_names == {"GoodModel", "AnotherGoodModel"}
    
    def test_schema_inference_empty_file(self, tmp_path):
        """Test handling of completely empty files."""
        file = tmp_path / "empty.py"
        file.write_text("")
        
        inferer = SchemaInferer(str(file))
        schemas = inferer.infer()
        
        # Should handle empty file gracefully
        assert schemas == []
    
    def test_schema_inference_whitespace_only_file(self, tmp_path):
        """Test handling of files containing only whitespace."""
        file = tmp_path / "whitespace.py"
        file.write_text("   \n\t\n   \n")
        
        inferer = SchemaInferer(str(file))
        schemas = inferer.infer()
        
        # Should handle whitespace-only files gracefully
        assert schemas == []


def test_dataclass_inference(tmp_path):
    file = tmp_path / "models.py"
    file.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            @dataclass
            class User:
                id: int
                name: str = 'anon'
            """
        )
    )
    schemas = SchemaInferer(str(file)).infer()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema.name == "User"
    assert len(schema.fields) == 2
    id_field = next(f for f in schema.fields if f.name == "id")
    assert id_field.type == "int"
    assert id_field.required
    name_field = next(f for f in schema.fields if f.name == "name")
    assert not name_field.required


def test_pydantic_basemodel_inference(tmp_path):
    file = tmp_path / "models.py"
    file.write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel

            class Item(BaseModel):
                id: int
                price: float = 0.0
            """
        )
    )
    schemas = SchemaInferer(str(file)).infer()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema.name == "Item"
    assert len(schema.fields) == 2
