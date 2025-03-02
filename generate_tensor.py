import numpy as np

def generate_sample_tensor(shape=(20, 20, 20, 20)):
    """Генерация демонстрационного тензора с шаблоном (для примера)"""
    print("Generating sample tensor...")
    tensor = np.zeros(shape, dtype=np.uint8)
    x = np.linspace(-3, 3, shape[0])
    y = np.linspace(-3, 3, shape[1])
    z = np.linspace(-3, 3, shape[2])
    w = np.linspace(-3, 3, shape[3])
    for i, xi in enumerate(x):
        print(f"Generating slice {i + 1}/{shape[0]}...", end="\r")
        for j, yi in enumerate(y):
            for k, zi in enumerate(z):
                distance = np.sqrt(xi ** 2 + yi ** 2 + zi ** 2 + w ** 2)
                pattern = np.exp(-distance)
                wave = np.sin(xi * yi) * np.cos(zi * w)
                combined = pattern + 0.5 * wave
                tensor[i, j, k, :] = np.clip((combined * 255), 0, 255).astype(np.uint8)
    print("\nSample tensor generated successfully!")
    return tensor

if __name__ == "__main__":
    # Генерация тензора
    tensor = generate_sample_tensor()
    # Сохранение тензора в файл
    np.save('sample_tensor.npy', tensor)
    print("Tensor saved to 'sample_tensor.npy'")