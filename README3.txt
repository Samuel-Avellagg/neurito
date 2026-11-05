Neurito 3.0: aprende a encontrar comida en un laberinto
Neurito busca comida dentro de tres laberintos. Su cerebro debe elegir entre arriba, abajo, izquierda y derecha. La red se construye con NumPy, y Pygame dibuja el escenario, la mascota y una gráfica de la pérdida.

Cómo aprende
BFS calcula rutas cortas y genera ejemplos etiquetados: mapa, posición, meta y movimiento recomendado.
Los ejemplos se dividen en entrenamiento (80 %) y prueba (20 %). Los pares posición-meta de prueba no aparecen en entrenamiento.
Cada entrada incluye un vector one-hot de tres mapas y dos vectores one-hot de 81 casillas: 165 características.
La arquitectura es 165 → 96 → 4. Usa ReLU en la capa oculta y Softmax en las cuatro salidas.
El entrenamiento minimiza entropía cruzada multiclase mediante retropropagación y descenso de gradiente con mini-lotes.
Durante el juego, el modelo decide sin consultar BFS. BFS es el experto que proporcionó las etiquetas.
Esto es aprendizaje supervisado por imitación, NO aprendizaje por refuerzo.

Dos modos
Red pura: toma la salida de mayor probabilidad. Puede chocar con paredes o quedarse en ciclos.
Asistido: descarta movimientos contra paredes y penaliza casillas visitadas. Son reglas adicionales, no conocimientos aprendidos por la red.
La precisión de clasificación no equivale al porcentaje de laberintos completados.

Estructura
laberinto_mundo.py: mapas, movimiento, BFS y navegación asistida.
laberinto_red.py: red neuronal y entrenamiento con NumPy.
laberinto.py: juego Pygame, sprites y pérdida.
test_laberinto.py: pruebas automáticas.
GUIA_LABERINTO.md: esta guía.
Reutiliza assets/neutral_1.png, neutral_2.png y neutral_3.png. No modifica las versiones anteriores.

Ejecutar
uv sync
uv run laberinto.py
Controles
Tecla	Acción
1	Entrenar una época
2	Entrenar cien épocas
3	Entrenamiento continuo
A	Alternar red pura y navegación asistida
P	Alternar navegación automática y paso a paso
Espacio	Avanzar un paso en modo manual
N	Cambiar mapa
G	Nuevas posiciones y meta
C	Reiniciar contadores
R	Reiniciar pesos y entrenamiento
Esc	Salir
Probar automáticamente
uv run python -m unittest -v test_laberinto.py
Experimentos
Observa Neurito sin entrenar. ¿Por qué se atasca?
Entrena 100, 200 y 400 épocas. Compara pérdida, precisión y movimientos.
Compara red pura frente a navegación asistida. ¿Qué parte corresponde al modelo y cuál a las reglas?
Cambia las 96 neuronas ocultas por 16, 32 o 128.
Compara la tasa por defecto 0.45 con 0.1 y 0.8; observa si la pérdida converge.
Cambia de mapa. Los mapas forman parte del entrenamiento, pero las combinaciones de inicio y meta usadas para probar no se entrenaron.
Compara DFS, BFS y A* como algoritmos expertos que generan las rutas; observa cómo cambian las rutas y el comportamiento aprendido.
Resultados de los experimentos
Método y verificación
Se usó el conjunto reproducible de generar_datos(semilla=42): 2,860 pares posición-meta para entrenamiento y 716 pares no vistos para prueba, repartidos entre los tres mapas. La red empezó con semilla 42; cada época entrenó con mini-lotes de 256. Se reportan entropía cruzada (pérdida) y exactitud sobre los 716 ejemplos de prueba.

Para medir el comportamiento se simuló cada par de prueba con el mismo límite de movimientos de las pruebas automáticas: min(65, 2 × distancia óptima + 8). El promedio de pasos se calcula solo entre los retos que llegaron a la comida. Las tasas corresponden a estos pares de prueba y no predicen resultados en mapas desconocidos.

La suite indicada arriba pasó: 5 pruebas correctas.

1 y 2. Sin entrenar y entrenamiento por épocas
La red pura usa solo el movimiento con mayor probabilidad. El modo asistido además evita paredes y penaliza casillas visitadas. La red empieza con pesos aleatorios; por eso sin entrenar su salida no representa una ruta aprendida.

Épocas	Pérdida de entrenamiento	Pérdida de prueba	Exactitud prueba	Éxito puro	Pasos* puro	Éxito asistido	Pasos* asistido
0	—	1.39403	27.65%	3.07%	1.82	57.54%	10.75
100	0.13521	0.36752	84.92%	51.26%	5.54	97.91%	7.88
200	0.03764	0.24205	90.08%	81.28%	6.01	98.88%	6.64
400	0.01066	0.22282	91.48%	82.68%	6.04	99.58%	6.59
*Pasos medios de los retos completados.

Sin entrenamiento, la red pura chocó con paredes en 589 de 716 retos y completó 22. La asistencia evitó todos esos choques, pero completó 412. A las 400 épocas la red pura completó 592 retos y la asistida 713; la asistencia aportó una mejora adicional al modelo.

3. Red pura frente a navegación asistida
En la red pura, 100 épocas dieron 51.26% de éxito y 15 intentos contra pared; a 400 épocas subió a 82.68%, con 3 intentos contra pared. El modo asistido nunca eligió una pared y llegó a 99.58% de éxito con 400 épocas. Esa mejora no es toda aprendizaje: el filtro de paredes y la penalización por visitas son reglas externas a la red.

4. Neuronas ocultas
Se mantuvieron la semilla, los datos, la tasa 0.45 y 400 épocas; solo cambió el tamaño de la capa oculta.

Neuronas	Pérdida entrenamiento	Pérdida prueba	Exactitud prueba	Éxito puro	Éxito asistido
16	0.03424	0.36190	87.15%	74.30%	99.02%
32	0.01608	0.29402	89.25%	78.35%	98.32%
96 (base)	0.01066	0.22282	91.48%	82.68%	99.58%
128	0.01028	0.25150	90.78%	85.34%	98.74%
Con 96 neuronas se obtuvo la mejor exactitud y el mejor resultado asistido de esta corrida. 128 dio el mayor éxito puro, aunque su exactitud fue algo menor. Más neuronas no mejoraron todas las métricas a la vez.

5. Tasa de aprendizaje
Se conservaron 96 neuronas y 400 épocas. Se comparó la tasa base 0.45 con 0.1 y 0.8.

Tasa	Pérdida entrenamiento	Pérdida prueba	Exactitud prueba	Éxito puro	Éxito asistido
0.1	0.14905	0.30945	87.15%	60.20%	98.88%
0.45 (base)	0.01066	0.22282	91.48%	82.68%	99.58%
0.8	0.00397	0.21990	93.02%	86.45%	99.02%
En 400 épocas, 0.1 aprendió más despacio. 0.8 tuvo la mejor exactitud y éxito puro de los valores probados, mientras 0.45 dio el mejor éxito asistido. La tasa más alta no fue mejor en cada medida; la convergencia también se debe comprobar en pérdida de prueba y recorridos.

6. Resultados por mapa
Con la configuración base (96 neuronas, 0.45, 400 épocas), los pares de prueba se agruparon por mapa.

Mapa	Retos	Éxito puro	Éxito asistido
1	210	85.71% (180)	100% (210)
2	251	78.49% (197)	98.80% (248)
3	255	84.31% (215)	100% (255)
La exactitud y el éxito variaron entre mapas. El mapa 2 fue el más difícil para esta red y también dejó tres retos sin resolver en modo asistido. Como los tres mapas aparecen en el entrenamiento, esto mide pares nuevos dentro de mapas conocidos, no generalización a un mapa ajeno.

7. DFS, BFS y A*
Se generaron etiquetas alternativas para los mismos 3,576 pares posición-meta y se mantuvo la misma división de 2,860/716. DFS y BFS usaron el mismo orden de movimientos; A* usó distancia Manhattan como heurística. Se entrenó cada red con 96 neuronas, tasa 0.45 y 400 épocas. Los nodos expandidos son el promedio por búsqueda hasta encontrar la meta.

Experto	Longitud media de ruta	Nodos expandidos medios	Rutas no óptimas frente a BFS	Exactitud de la red	Éxito puro	Éxito asistido
BFS	6.02 pasos	18.56	0%	91.48%	82.68%	99.58%
DFS	11.15 pasos	18.56	61.77%	94.83%	11.45%	35.75%
A*	6.02 pasos	8.27	0%	92.60%	86.45%	99.30%
BFS y A* encontraron rutas óptimas en estos mapas; A* expandió aproximadamente 55% menos nodos que BFS. DFS encontró rutas más largas en 61.77% de los pares. Aunque la red imitadora de DFS alcanzó 94.83% de exactitud sobre sus etiquetas, solo completó 11.45% de los retos en modo puro. La red toma decisiones paso a paso: una decisión correcta para cada ejemplo aislado no garantiza que la secuencia forme una ruta coherente.

A* mejoró modestamente el éxito puro frente a BFS en esta corrida (86.45% frente a 82.68%); en modo asistido ambos quedaron cerca del 99%. El resultado depende de estos mapas, desempates, semillas y pares probados.

Conclusión
Los resultados muestran por qué hay que medir tanto la clasificación como la llegada a la meta. Las reglas de asistencia evitan paredes, pero no reemplazan una ruta aprendida; DFS puede producir etiquetas que se imitan bien localmente y se comportan mal como política secuencial. BFS conserva la referencia de camino corto y A* encuentra esas mismas longitudes con menos expansiones en estos mapas.

Alcance y limitaciones
El modelo no interpreta capturas de pantalla: trabaja con características estructuradas. Conoce el identificador de cada mapa. Los tres mapas pertenecen al mismo universo de entrenamiento. La evaluación demuestra generalización a pares de posición-meta no entrenados, no a laberintos totalmente desconocidos. Para eso habría que representar también las paredes de cualquier mapa nuevo y entrenar y evaluar en mapas separados.
