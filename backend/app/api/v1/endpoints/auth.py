"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from app.core.security import create_access_token, create_refresh_token, hash_password, verify_password

router = APIRouter()

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class UserRegister(BaseModel):
    email: str
    password: str
    full_name: str = ""

@router.post("/login", response_model=TokenResponse, summary="Authenticate user")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # In production, look up user from DB and verify password
    # For demo: accept any credentials
    if not form_data.username or not form_data.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token_data = {"sub": form_data.username, "email": form_data.username, "role": "admin"}
    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
    )

@router.post("/register", response_model=TokenResponse, summary="Register new user")
async def register(user: UserRegister):
    hashed = hash_password(user.password)
    token_data = {"sub": user.email, "email": user.email, "role": "viewer"}
    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
    )

@router.post("/refresh", summary="Refresh access token")
async def refresh_token(refresh_token: str):
    from app.core.security import decode_token
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    token_data = {"sub": payload["sub"], "email": payload.get("email", ""), "role": payload.get("role", "viewer")}
    return {"access_token": create_access_token(token_data), "token_type": "bearer"}
