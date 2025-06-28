from fastapi import FastAPI

app = FastAPI()


@app.get("/hello")
def hello() -> dict[str, str]:
    return {"message": "Hello world"}
