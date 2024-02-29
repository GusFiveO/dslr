#!/usr/bin/env python3

import subprocess

def install_requirements():
    try:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        print("Les bibliothèques ont été installées avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Une erreur s'est produite lors de l'installation des bibliothèques: {e}")

if __name__ == "__main__":
    install_requirements()