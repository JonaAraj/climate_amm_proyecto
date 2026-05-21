@echo off
echo Instalando PyInstaller y asegurando dependencias...
pip install pyinstaller

echo Construyendo el ejecutable (esto puede tomar varios minutos)...

:: Utilizamos --onedir en lugar de --onefile por cuestiones de rendimiento,
:: ya que desempaquetar librerias pesadas como SciPy y Numpy cada vez que abres
:: un --onefile hace que el programa tarde mucho en iniciar.

pyinstaller --noconfirm --onedir ^
  --name "Pronostico_AMM" ^
  --add-data "dashboard.py;." ^
  --add-data "modules;modules" ^
  --copy-metadata streamlit ^
  --hidden-import "plotly" ^
  --hidden-import "scipy.integrate" ^
  --hidden-import "scipy.stats" ^
  run_app.py

echo Proceso completado. El programa se encuentra en la carpeta 'dist\Pronostico_AMM'.
pause