
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # Powinno pokazać GTX 1650

import torch
import time

# Wybór GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tworzymy duże tensory
size = 10000  # 10k x 10k
print("Tworzenie tensorów...")
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Proste mnożenie macierzy
print("Mnożenie macierzy na GPU...")
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()  # upewniamy się, że wszystkie operacje się zakończyły
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")

# Sprawdzenie GPU w nvidia-smi w trakcie działania
print("Sprawdź teraz użycie GPU w innym terminalu: nvidia-smi")
