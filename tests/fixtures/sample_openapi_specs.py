"""Sample OpenAPI specifications for testing."""

BASIC_OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Test API",
        "version": "1.0.0",
        "description": "A test API specification"
    },
    "paths": {
        "/api/users": {
            "get": {
                "summary": "List users",
                "responses": {
                    "200": {
                        "description": "List of users",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/User"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "summary": "Create user",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/UserCreate"}
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "User created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/User"}
                            }
                        }
                    }
                }
            }
        },
        "/api/users/{user_id}": {
            "get": {
                "summary": "Get user by ID",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "User details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/User"}
                            }
                        }
                    },
                    "404": {
                        "description": "User not found"
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["id", "name", "email"]
            },
            "UserCreate": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["name", "email"]
            }
        }
    }
}

COMPLEX_OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "E-commerce API",
        "version": "2.1.0",
        "description": "Comprehensive e-commerce API with authentication"
    },
    "servers": [
        {"url": "https://api.example.com/v2", "description": "Production server"},
        {"url": "https://staging.api.example.com/v2", "description": "Staging server"}
    ],
    "security": [
        {"bearerAuth": []}
    ],
    "paths": {
        "/products": {
            "get": {
                "summary": "List products",
                "parameters": [
                    {
                        "name": "category",
                        "in": "query",
                        "schema": {"type": "string"}
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer", "minimum": 1, "maximum": 100}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Product list",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "products": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/Product"}
                                        },
                                        "pagination": {"$ref": "#/components/schemas/Pagination"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/orders": {
            "post": {
                "summary": "Create order",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/OrderCreate"}
                        }
                    }
                },
                "responses": {
                    "201": {"description": "Order created"},
                    "400": {"description": "Invalid request"},
                    "401": {"description": "Unauthorized"}
                }
            }
        }
    },
    "components": {
        "schemas": {
            "Product": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "price": {"type": "number", "format": "float"},
                    "category": {"type": "string"},
                    "in_stock": {"type": "boolean"}
                }
            },
            "OrderCreate": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {"type": "integer"},
                                "quantity": {"type": "integer", "minimum": 1}
                            }
                        }
                    },
                    "shipping_address": {"$ref": "#/components/schemas/Address"}
                }
            },
            "Address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "postal_code": {"type": "string"}
                }
            },
            "Pagination": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "per_page": {"type": "integer"},
                    "total": {"type": "integer"}
                }
            }
        },
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
    }
}

INVALID_OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Invalid API"
        # Missing version field
    },
    "paths": {
        "/test": {
            "get": {
                # Missing responses field
            }
        }
    }
}