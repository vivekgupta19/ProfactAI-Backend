import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
from typing import Optional, Dict, Any
import re

from database import db

JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.environ.get("JWT_EXPIRATION_HOURS", "24"))


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def generate_token(user_id: str) -> str:
    """Generate a JWT token for a user"""
    expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "userId": user_id,
        "exp": expiration,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify a JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, ""


def create_slug(text: str) -> str:
    """Create a URL-friendly slug from text"""
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug


def get_current_user_from_request() -> Optional[Dict[str, Any]]:
    """Extract and validate user from request authorization header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    session = db.get_session_by_token(token)
    
    if not session:
        return None
    
    return {
        "userId": session["userid"],
        "email": session["email"],
        "fullName": session["fullname"],
        "role": session["role"],
        "organizationId": session["organizationid"],
        "organizationName": session["organizationname"],
        "organizationSlug": session["organizationslug"]
    }


def require_auth(f):
    """Decorator to require authentication for a route"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user_from_request()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        return f(user, *args, **kwargs)
    return decorated_function


def require_role(*allowed_roles):
    """Decorator to require specific roles for a route"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = get_current_user_from_request()
            if not user:
                return jsonify({"error": "Authentication required"}), 401
            if user["role"] not in allowed_roles:
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(user, *args, **kwargs)
        return decorated_function
    return decorator


def register_user(
    email: str,
    password: str,
    full_name: str,
    organization_name: str
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Register a new user and organization
    Returns: (user_data, error_message)
    """
    # Validate inputs
    if not validate_email(email):
        return None, "Invalid email format"
    
    is_valid, error_msg = validate_password(password)
    if not is_valid:
        return None, error_msg
    
    if not full_name or len(full_name.strip()) < 2:
        return None, "Full name must be at least 2 characters"
    
    if not organization_name or len(organization_name.strip()) < 2:
        return None, "Organization name must be at least 2 characters"
    
    # Check if user already exists
    existing_user = db.get_user_by_email(email)
    if existing_user:
        return None, "User with this email already exists"
    
    # Create organization slug
    org_slug = create_slug(organization_name)
    
    # Check if organization slug exists, append number if needed
    base_slug = org_slug
    counter = 1
    while db.get_organization_by_slug(org_slug):
        org_slug = f"{base_slug}-{counter}"
        counter += 1
    
    try:
        # Create organization
        organization = db.create_organization(organization_name, org_slug)
        
        # Hash password and create user
        password_hash = hash_password(password)
        user = db.create_user(
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            organization_id=organization["organizationid"],
            role="admin"  # First user in organization is admin
        )
        
        return {
            "userId": user["userid"],
            "email": user["email"],
            "fullName": user["fullname"],
            "role": user["role"],
            "organizationId": organization["organizationid"],
            "organizationName": organization["organizationname"],
            "organizationSlug": organization["organizationslug"]
        }, None
    except Exception as e:
        return None, f"Failed to create user: {str(e)}"


def login_user(email: str, password: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Authenticate user and create session
    Returns: (auth_data, error_message)
    """
    if not email or not password:
        return None, "Email and password are required"
    
    # Get user from database
    user = db.get_user_by_email(email)
    if not user:
        return None, "Invalid email or password"
    
    # Verify password
    if not verify_password(password, user["passwordhash"]):
        return None, "Invalid email or password"
    
    try:
        # Generate token
        token = generate_token(user["userid"])
        
        # Create session
        expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        db.create_session(user["userid"], token, expiration)
        
        # Update last login
        db.update_last_login(user["userid"])
        
        return {
            "token": token,
            "user": {
                "userId": user["userid"],
                "email": user["email"],
                "fullName": user["fullname"],
                "role": user["role"],
                "organizationId": user["organizationid"],
                "organizationName": user["organizationname"],
                "organizationSlug": user["organizationslug"]
            }
        }, None
    except Exception as e:
        return None, f"Failed to create session: {str(e)}"


def logout_user(token: str) -> bool:
    """Logout user by deleting session"""
    try:
        db.delete_session(token)
        return True
    except:
        return False