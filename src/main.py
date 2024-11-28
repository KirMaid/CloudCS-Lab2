# -*- coding: utf-8 -*-
import os
from model_utils import load_model, make_inference
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi_utils import Oauth2ClientCredentials
from pydantic import BaseModel
from keycloak.uma_permissions import AuthStatus
from keycloak_utils import get_keycloak_data
import requests


class Instance(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: int
    island_Biscoe: int
    island_Dream: int
    island_Torgersen: int


class Credentials(BaseModel):
    client_id: str
    client_secret: str


app = FastAPI()
keycloak_openid, token_endpoint = get_keycloak_data()

model_path: str = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("The environment variable $MODEL_PATH is empty!")


async def get_token_status(token: str) -> AuthStatus:
    return keycloak_openid.has_uma_access(
        token, "infer_endpoint#doInfer")


async def get_access_token(credentials: Credentials):
    data = {
        'grant_type': 'client_credentials',
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    response = requests.post(token_endpoint, data=data,
                             headers=headers, verify=False)

    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Failed to obtain access token: {response.text}"
        )


@app.post("/predictions")
async def predictions(instance: Instance, credentials: Credentials) -> dict[str, str]:
    token = await get_access_token(credentials)
    auth_status = await get_token_status(token)

    is_logged = auth_status.is_logged_in
    is_authorized = auth_status.is_authorized
    print("--------")
    print(auth_status)
    print("--------")
    if not is_logged:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials :()",
            headers={"WWW-Authenticate": "Bearer"},
        )
    elif not is_authorized:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return make_inference(load_model(model_path),
                          instance.dict())


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    print(token_endpoint)
    return {"status": "ok"}
