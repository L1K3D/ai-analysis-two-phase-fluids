import sys
import subprocess

def install_libs(libs):
    """
    Run pip install silently for the list of libraries.
    If they are already installed, pip simply ignores them and moves on quickly.
    """
    print("Checking project dependencies...")
    for lib in libs:
        try:
            # sys.executable garante que o pip usado é o mesmo do Python atual
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', lib])
            print(f"✔️ {lib} pronta.")
        except subprocess.CalledProcessError:
            print(f"❌ Falha ao tentar instalar {lib}.")


def install_missing_libs(libs):
    """Compatibility name used by notebooks for installing libraries."""
    install_libs(libs)