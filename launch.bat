@echo off
echo ========================================
echo Lancement RaGME_UP - PROP
echo ========================================
echo.

REM Vérifier que Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installé ou n'est pas dans le PATH
    echo.
    echo Veuillez d'abord executer install.bat
    echo.
    pause
    exit /b 1
)

REM Vérifier que Streamlit est installé
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Streamlit n'est pas installe
    echo.
    echo Veuillez d'abord executer install.bat
    echo.
    pause
    exit /b 1
)

REM Vérifier que le fichier principal existe
if not exist "streamlit_RAG.py" (
    echo [ERREUR] Le fichier streamlit_RAG.py est introuvable
    echo.
    echo Assurez-vous d'executer ce fichier depuis le bon repertoire
    echo.
    pause
    exit /b 1
)

echo [OK] Demarrage de l'application...
echo.
echo L'application va s'ouvrir dans votre navigateur par defaut.
echo Si ce n'est pas le cas, ouvrez manuellement : http://localhost:8501
echo.
echo Pour arreter l'application : fermez cette fenetre ou appuyez sur Ctrl+C
echo.
echo ========================================
echo.

REM Lancer Streamlit
streamlit run streamlit_RAG.py

REM Si Streamlit se termine, attendre avant de fermer
echo.
echo L'application s'est arretee.
pause
