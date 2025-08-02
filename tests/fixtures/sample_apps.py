"""Sample application code for testing framework discovery."""

FLASK_APP_BASIC = '''
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/api/users", methods=["GET", "POST"])
def users():
    """User management endpoint."""
    if request.method == "GET":
        return jsonify({"users": []})
    return jsonify({"message": "User created"}), 201

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})
'''

FLASK_APP_COMPLEX = '''
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class UserResource(Resource):
    """User resource with full CRUD operations."""
    
    def get(self, user_id=None):
        """Get user(s)."""
        if user_id:
            return {"user": {"id": user_id, "name": "John"}}
        return {"users": []}
    
    def post(self):
        """Create new user."""
        data = request.get_json()
        return {"user": data}, 201
    
    def put(self, user_id):
        """Update user."""
        data = request.get_json()
        return {"user": {"id": user_id, **data}}
    
    def delete(self, user_id):
        """Delete user."""
        return "", 204

api.add_resource(UserResource, "/api/users", "/api/users/<int:user_id>")
'''

FASTAPI_APP_BASIC = '''
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Test API", version="1.0.0")

class User(BaseModel):
    id: int
    name: str
    email: str

class UserCreate(BaseModel):
    name: str
    email: str

@app.get("/api/users", response_model=List[User])
async def get_users():
    """Get all users."""
    return []

@app.post("/api/users", response_model=User)
async def create_user(user: UserCreate):
    """Create a new user."""
    return User(id=1, **user.dict())

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
'''

FASTAPI_APP_COMPLEX = '''
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

app = FastAPI(
    title="E-commerce API",
    description="Advanced e-commerce API with authentication",
    version="2.0.0"
)

security = HTTPBearer()

class CategoryEnum(str, Enum):
    electronics = "electronics"
    clothing = "clothing"
    books = "books"

class Product(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    category: CategoryEnum
    in_stock: bool = True

class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    category: CategoryEnum

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    return {"user_id": 1, "username": "test"}

@app.get("/products", response_model=List[Product])
async def list_products(
    category: Optional[CategoryEnum] = None,
    limit: int = Field(10, ge=1, le=100)
):
    """List products with filtering."""
    return []

@app.post("/products", response_model=Product)
async def create_product(
    product: ProductCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new product (authenticated)."""
    return Product(id=1, **product.dict())
'''

DJANGO_URLS = '''
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'products', views.ProductViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('health/', views.health_check, name='health'),
    path('api/auth/login/', views.login, name='login'),
]
'''

DJANGO_VIEWS = '''
from django.http import JsonResponse
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import User, Product
from .serializers import UserSerializer, ProductSerializer

class UserViewSet(viewsets.ModelViewSet):
    """User management viewset."""
    queryset = User.objects.all()
    serializer_class = UserSerializer

class ProductViewSet(viewsets.ReadOnlyModelViewSet):
    """Product listing viewset."""
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

@api_view(['GET'])
def health_check(request):
    """Health check endpoint."""
    return Response({"status": "ok"})

@api_view(['POST'])
def login(request):
    """User login endpoint."""
    return Response({"token": "fake-jwt-token"})
'''

EXPRESS_APP_BASIC = '''
const express = require('express');
const app = express();

app.use(express.json());

/**
 * Get all users
 * @route GET /api/users
 * @returns {Array} List of users
 */
app.get('/api/users', (req, res) => {
    res.json({ users: [] });
});

/**
 * Create new user
 * @route POST /api/users
 * @param {Object} user - User data
 * @returns {Object} Created user
 */
app.post('/api/users', (req, res) => {
    res.status(201).json({ user: req.body });
});

/**
 * Health check endpoint
 * @route GET /health
 * @returns {Object} Health status
 */
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

module.exports = app;
'''

TORNADO_APP_BASIC = '''
import tornado.web
import json

class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")

class UsersHandler(BaseHandler):
    """User management handler."""
    
    def get(self):
        """Get all users."""
        self.write({"users": []})
    
    def post(self):
        """Create new user."""
        data = json.loads(self.request.body)
        self.set_status(201)
        self.write({"user": data})

class UserHandler(BaseHandler):
    """Individual user handler."""
    
    def get(self, user_id):
        """Get user by ID."""
        self.write({"user": {"id": int(user_id), "name": "John"}})

class HealthHandler(BaseHandler):
    """Health check handler."""
    
    def get(self):
        """Health check endpoint."""
        self.write({"status": "ok"})

def make_app():
    return tornado.web.Application([
        (r"/api/users", UsersHandler),
        (r"/api/users/([0-9]+)", UserHandler),
        (r"/health", HealthHandler),
    ])
'''

GRAPHQL_SCHEMA = '''
type User {
  id: ID!
  name: String!
  email: String!
  posts: [Post!]!
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  createdAt: String!
}

type Query {
  users: [User!]!
  user(id: ID!): User
  posts: [Post!]!
  post(id: ID!): Post
}

type Mutation {
  createUser(name: String!, email: String!): User!
  updateUser(id: ID!, name: String, email: String): User!
  deleteUser(id: ID!): Boolean!
  
  createPost(title: String!, content: String!, authorId: ID!): Post!
  updatePost(id: ID!, title: String, content: String): Post!
  deletePost(id: ID!): Boolean!
}

type Subscription {
  userCreated: User!
  postCreated: Post!
}
'''