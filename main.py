
from pathlib import Path

import numpy as np
import pygame

# ============================================================
# CONFIGURACION DE LOS EXPERIMENTOS
# Cambia estos valores antes de cada ejecucion.
# ============================================================

LANGUAGE = "es"

HIDDEN_NEURONS = 6       # Probar: 3, 6, 10, 20
EPOCHS = 1200            # Probar: 100, 1200, 3000
LEARNING_RATE = 0.1      # Probar: 0.001, 0.1, 1.0

SEED = 42

TEXT = {
    "en": {
        "title": "NEURITO - MY ARTIFICIAL BRAIN",
        "hunger": "Hunger", "energy": "Energy",
        "boredom": "Boredom",
        "eat": "EAT", "sleep": "SLEEP",
        "play": "PLAY", "neutral": "CALM",
        "accuracy": "Test accuracy",
        "epoch": "Epoch", "loss": "Loss",
        "controls1": "H/J: hunger +/-  E/D: energy +/-",
        "controls2": "A/S: boredom +/-  R: random",
        "controls3": "N: reset  ESC: quit",
    },
    "es": {
        "title": "NEURITO - MI CEREBRO ARTIFICIAL",
        "hunger": "Hambre", "energy": "Energia",
        "boredom": "Aburrimiento",
        "eat": "COMER", "sleep": "DORMIR",
        "play": "JUGAR", "neutral": "TRANQUILO",
        "accuracy": "Precision de prueba",
        "epoch": "Epoca", "loss": "Perdida",
        "controls1": "H/J: hambre +/-  E/D: energia +/-",
        "controls2": "A/S: aburrimiento +/-  R: aleatorio",
        "controls3": "N: reiniciar  ESC: salir",
    },
}

labels = TEXT[LANGUAGE]

# La cuarta salida es NEUTRAL, aprendida por la red.
actions = ("eat", "sleep", "play", "neutral")

# ============================================================
# DATOS SINTETICOS
# Entradas: hambre, energia y aburrimiento.
# Salidas: comer, dormir, jugar y estado neutral.
# ============================================================

np.random.seed(SEED)

X = np.random.rand(3000, 3)

hunger = X[:, 0]
energy = X[:, 1]
boredom = X[:, 2]
tiredness = 1 - energy

# Puntuaciones de la politica sintetica.
# El estado neutral se favorece cuando las tres necesidades
# son bajas y la energia es alta.
calm_score = np.minimum.reduce([
    1 - hunger,
    energy,
    1 - boredom
])

scores = np.column_stack((
    hunger,
    tiredness,
    boredom,
    calm_score
))

targets = np.argmax(scores, axis=1)

# Inicializacion de los pesos y sesgos.
W1 = np.random.randn(3, HIDDEN_NEURONS) * 0.5
b1 = np.zeros((1, HIDDEN_NEURONS))

W2 = np.random.randn(HIDDEN_NEURONS, len(actions)) * 0.5
b2 = np.zeros((1, len(actions)))


# ============================================================
# FUNCIONES DE LA RED NEURONAL
# ============================================================

def relu(values):
    return np.maximum(0, values)


def softmax(values):
    exp_values = np.exp(
        values - np.max(values, axis=1, keepdims=True)
    )
    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )


def predict(data):
    hidden = relu(data @ W1 + b1)
    return softmax(hidden @ W2 + b2)


def calculate_loss(probabilities, expected):
    count = len(expected)
    return -np.mean(
        np.log(
            probabilities[np.arange(count), expected] + 1e-10
        )
    )


# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\n--- CONFIGURACION DEL EXPERIMENTO ---")
print(f"Neuronas ocultas: {HIDDEN_NEURONS}")
print(f"Epocas: {EPOCHS}")
print(f"Tasa de aprendizaje: {LEARNING_RATE}")

for epoch in range(EPOCHS):
    z1 = X @ W1 + b1
    hidden = relu(z1)

    probabilities = softmax(hidden @ W2 + b2)
    count = len(X)

    loss = calculate_loss(probabilities, targets)

    delta2 = probabilities.copy()
    delta2[np.arange(count), targets] -= 1
    delta2 /= count

    grad_W2 = hidden.T @ delta2
    grad_b2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = (delta2 @ W2.T) * (z1 > 0)

    grad_W1 = X.T @ delta1
    grad_b1 = np.sum(delta1, axis=0, keepdims=True)

    W1 -= LEARNING_RATE * grad_W1
    b1 -= LEARNING_RATE * grad_b1
    W2 -= LEARNING_RATE * grad_W2
    b2 -= LEARNING_RATE * grad_b2

    if epoch % 200 == 0 or epoch == EPOCHS - 1:
        print(
            f"{labels['epoch']} {epoch + 1} | "
            f"{labels['loss']}: {loss:.4f}"
        )

# ============================================================
# EVALUACION CON DATOS NUEVOS
# ============================================================

test = np.random.rand(1000, 3)

test_scores = np.column_stack((
    test[:, 0],
    1 - test[:, 1],
    test[:, 2],
    np.minimum.reduce((
        1 - test[:, 0],
        test[:, 1],
        1 - test[:, 2]
    ))
))

test_targets = np.argmax(test_scores, axis=1)
test_probabilities = predict(test)

accuracy = np.mean(
    np.argmax(test_probabilities, axis=1) == test_targets
)

test_loss = calculate_loss(test_probabilities, test_targets)

print("\n--- RESULTADOS FINALES ---")
print(f"Perdida de prueba: {test_loss:.4f}")
print(f"Precision de prueba: {accuracy:.2%}")
print(f"Neuronas ocultas: {HIDDEN_NEURONS}")
print(f"Epocas: {EPOCHS}")
print(f"Tasa de aprendizaje: {LEARNING_RATE}")

# ============================================================
# INTERFAZ GRAFICA CON PYGAME
# ============================================================

pygame.init()

screen = pygame.display.set_mode((900, 650))
pygame.display.set_caption("Neurito - Mi cerebro artificial")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
small_font = pygame.font.SysFont(None, 22)

asset_dir = Path(__file__).resolve().parent / "assets"
pet_size = (300, 300)


def load_frames(names):
    result = []

    for name in names:
        path = asset_dir / name
        image = pygame.image.load(str(path)).convert_alpha()
        image = pygame.transform.smoothscale(image, pet_size)
        result.append(image)

    return result


frames = {
    "neutral": load_frames([
        "neutral_1.png", "neutral_2.png", "neutral_3.png"
    ]),
    "eat": load_frames(["comer_1.png", "comer_2.png"]),
    "sleep": load_frames(["dormir_1.png", "dormir_2.png"]),
    "play": load_frames(["jugar_1.png", "jugar_2.png"]),
}

needs = [0.2, 0.9, 0.2]
current_state = "neutral"
frame_index = 0
last_frame_time = pygame.time.get_ticks()
running = True

colors = [
    (235, 125, 75),
    (70, 175, 115),
    (105, 125, 225)
]

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_h:
                needs[0] = min(1, needs[0] + 0.1)
            elif event.key == pygame.K_j:
                needs[0] = max(0, needs[0] - 0.1)

            elif event.key == pygame.K_e:
                needs[1] = min(1, needs[1] + 0.1)
            elif event.key == pygame.K_d:
                needs[1] = max(0, needs[1] - 0.1)

            elif event.key == pygame.K_a:
                needs[2] = min(1, needs[2] + 0.1)
            elif event.key == pygame.K_s:
                needs[2] = max(0, needs[2] - 0.1)

            elif event.key == pygame.K_r:
                needs = list(np.random.rand(3))

            elif event.key == pygame.K_n:
                needs = [0.2, 0.9, 0.2]

    probabilities = predict(np.array([needs]))[0]
    action = actions[int(np.argmax(probabilities))]
    state = action

    now = pygame.time.get_ticks()

    if state != current_state:
        current_state = state
        frame_index = 0
        last_frame_time = now
    elif now - last_frame_time >= 500:
        frame_index = (frame_index + 1) % len(frames[state])
        last_frame_time = now

    screen.fill((250, 245, 252))

    title = font.render(labels["title"], True, (55, 45, 75))
    screen.blit(title, (35, 25))

    # Mostrar necesidades.
    for index, key in enumerate(("hunger", "energy", "boredom")):
        value = needs[index]
        y = 115 + index * 85

        text = small_font.render(
            f"{labels[key]}: {value * 100:.0f}%",
            True, (35, 40, 55)
        )
        screen.blit(text, (45, y - 25))

        pygame.draw.rect(
            screen, (215, 220, 230),
            (45, y, 290, 20), border_radius=7
        )
        pygame.draw.rect(
            screen, colors[index],
            (45, y, int(290 * value), 20), border_radius=7
        )

    # Mostrar personaje y accion predicha.
    pet = frames[state][frame_index]
    screen.blit(pet, pet.get_rect(center=(650, 230)))

    screen.blit(
        font.render(labels[state], True, (65, 50, 85)),
        (570, 410)
    )

    # Mostrar las cuatro probabilidades de Softmax.
    screen.blit(
        small_font.render("Probabilidades Softmax:", True, (55, 50, 75)),
        (45, 385)
    )

    for index, action_name in enumerate(actions):
        score = small_font.render(
            f"{labels[action_name]}: "
            f"{probabilities[index] * 100:.2f}%",
            True, (55, 50, 75)
        )
        screen.blit(score, (45, 415 + index * 27))

    # Controles.
    screen.blit(
        small_font.render(labels["controls1"], True, (70, 70, 90)),
        (400, 510)
    )
    screen.blit(
        small_font.render(labels["controls2"], True, (70, 70, 90)),
        (400, 540)
    )
    screen.blit(
        small_font.render(labels["controls3"], True, (70, 70, 90)),
        (400, 570)
    )

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
