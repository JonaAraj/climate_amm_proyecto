import os
import sys
import streamlit.web.cli as stcli

def main():
    """
    Punto de entrada para el ejecutable compilado con PyInstaller.
    Configura las rutas dinámicas y lanza Streamlit programáticamente.
    """
    # Detectar si estamos corriendo en un bundle compilado (ejecutable)
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        # Ejecución normal en Python
        base_dir = os.path.dirname(os.path.abspath(__file__))

    dashboard_path = os.path.join(base_dir, 'dashboard.py')

    # Configurar los argumentos que normalmente pasarías por terminal
    sys.argv = [
        "streamlit",
        "run",
        dashboard_path,
        "--global.developmentMode=false"
    ]

    sys.exit(stcli.main())

if __name__ == "__main__":
    main()