import subprocess
import sys

def upgrade_torch():
    print("--- ROZPOCZYNAM NAPRAWĘ ŚRODOWISKA ---")
    
    # Pełna komenda wymuszająca wersję 2.6.0 z obsługą Twojej karty RTX 4070 Ti SUPER
    command = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "torch==2.6.0", "torchvision", "torchaudio",
        "--extra-index-url", "https://download.pytorch.org/whl/cu124"
    ]

    try:
        # Uruchomienie instalacji
        subprocess.run(command, check=True)
        print("\n--- SUKCES: PyTorch zaktualizowany do 2.6.0 ---")
        
        # Weryfikacja po instalacji
        import torch
        print(f"Zainstalowana wersja: {torch.__version__}")
        print(f"Obsługa CUDA: {torch.cuda.is_available()}")
        
    except Exception as e:
        print(f"\nBŁĄD: Nie udało się automatycznie zaktualizować. Spróbuj zamknąć wszystkie programy używające Pythona i uruchom skrypt ponownie.")
        print(f"Szczegóły: {e}")

if __name__ == "__main__":
    upgrade_torch()