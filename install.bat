@echo off
echo ========================================
echo Installation RaGME_UP - PROP
echo ========================================
echo.

REM Vérifier que Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installé ou n'est pas dans le PATH
    echo.
    echo Veuillez installer Python 3.8 ou supérieur depuis https://www.python.org/downloads/
    echo N'oubliez pas de cocher "Add Python to PATH" lors de l'installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python detecte :
python --version
echo.

REM Vérifier que pip est installé
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] pip n'est pas installé
    echo.
    echo Essayez d'installer pip avec : python -m ensurepip --upgrade
    echo.
    pause
    exit /b 1
)

echo [OK] pip detecte :
pip --version
echo.

REM Mettre à jour pip
echo [ETAPE 1/3] Mise a jour de pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ATTENTION] La mise a jour de pip a echoue, mais on continue...
)
echo.

REM Installer les dépendances principales
echo [ETAPE 2/3] Installation des dependances principales...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERREUR] L'installation des dependances a echoue
    echo Verifiez votre connexion internet et reessayez
    echo.
    pause
    exit /b 1
)
echo.

REM Installer les dépendances pour la GUI (CustomTkinter)
echo [ETAPE 3/3] Installation des dependances GUI...
pip install customtkinter pillow
if errorlevel 1 (
    echo.
    echo [ATTENTION] L'installation de CustomTkinter a echoue
    echo La GUI de gestion CSV pourrait ne pas fonctionner
    echo.
)
echo.

echo ========================================
echo Installation terminee avec succes !
echo ========================================
echo.
echo Vous pouvez maintenant lancer l'application avec : launch.bat
echo.
pause
