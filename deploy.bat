@echo off
setlocal

:: Cấu hình
set IMAGE_NAME=rl-robot-api
set DOCKER_USER=thanglele
set TAG=latest

echo === Build Docker image ===
docker build -t %DOCKER_USER%/%IMAGE_NAME%:%TAG% .

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)

echo === Push image to Docker Hub ===
docker push %DOCKER_USER%/%IMAGE_NAME%:%TAG%

if %errorlevel% neq 0 (
    echo Push failed!
    exit /b %errorlevel%
)

echo === Run container ===
@REM docker run -d --name %IMAGE_NAME% -p 8000:8000 %DOCKER_USER%/%IMAGE_NAME%:%TAG%

docker compose -f docker-compose.yml up -d

if %errorlevel% neq 0 (
    echo Run container failed!
    exit /b %errorlevel%
)

echo === Done! Container is running at http://localhost:8000 ===
endlocal

pause